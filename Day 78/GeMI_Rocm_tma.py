import os
import torch
import triton
import triton.language as tl
from typing import Tuple
input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
output_t = torch.Tensor

os.environ['ENABLE_TMA'] = '1'

def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    a = a.contiguous(); a_scale = a_scale.contiguous(); b_scale = b_scale.contiguous()
    m, k = a.shape; n = b.shape[0]
    block_shape_n, block_shape_k = 128, 128
    scale_n, scale_k = b_scale.shape
    a_scale_exp = a_scale.unsqueeze(-1).expand(m, scale_k, block_shape_k)
    a_scale_exp = a_scale_exp.reshape(m, scale_k * block_shape_k)[:, :k]
    a_deq = a.to(a_scale_exp.dtype) * a_scale_exp
    b_scale_r = b_scale.view(scale_n, scale_k, 1, 1)
    b_scale_exp = (b_scale_r
                   .expand(scale_n, scale_k, block_shape_n, block_shape_k)
                   .permute(0,2,1,3)
                   .reshape(scale_n*block_shape_n, scale_k*block_shape_k))[:n,:k]
    b_deq = b.to(b_scale_exp.dtype) * b_scale_exp
    tmp = torch.matmul(a_deq, b_deq.t())
    c.copy_(tmp.to(torch.bfloat16))
    return c

@triton.autotune(
    configs=[
        # Basic configurations
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 8}),
        # Larger BK
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'num_stages': 2, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 2, 'num_warps': 8}),
        # More stages (pipelining)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K'], 
)
@triton.jit
def _kernel(
    a_ptr, a_scl_ptr, b_ptr, b_scl_ptr, c_ptr,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    # Assuming scale strides match data strides or are handled by broadcasting before call
    stride_a_scl_m, stride_a_scl_k, stride_b_scl_n, stride_b_scl_k, # Pass scale strides
    scale_a: tl.constexpr, scale_b: tl.constexpr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ATOMIC: tl.constexpr, # Flag to choose between store and atomic_add
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        cur_offs_k = k_start + offs_k
        k_mask = cur_offs_k < K

        # --- Load A block and scale ---
        a_block_ptr = a_ptr + (offs_m[:, None] * stride_am + cur_offs_k[None, :] * stride_ak)
        # Assuming a_scale is compatible for element-wise load based on data block offsets
        a_scl_block_ptr = a_scl_ptr + (offs_m[:, None] * stride_a_scl_m + cur_offs_k[None, :] * stride_a_scl_k)

        a_block = tl.load(a_block_ptr, mask=k_mask[None, :], other=0.0, cache_modifier=".ca")
        # Load scale, default=1.0 if mask fails. Use scale strides.
        a_scl = tl.load(a_scl_block_ptr, mask=k_mask[None, :], other=1.0, cache_modifier=".ca")

        # --- Load B block and scale ---
        # B is N x K, access needs K x N pattern: rows=cur_offs_k, cols=offs_n
        b_block_ptr = b_ptr + (cur_offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        # Assuming b_scale is compatible for element-wise load based on data block offsets
        # Using b's N x K strides (stride_b_scl_n, stride_b_scl_k)
        b_scl_block_ptr = b_scl_ptr + (offs_n[None, :] * stride_b_scl_n + cur_offs_k[:,None] * stride_b_scl_k) # Adjusted for B scale layout relative to B data access


        b_block = tl.load(b_block_ptr, mask=k_mask[:, None], other=0.0, cache_modifier=".ca")
        # Load scale, default=1.0. Use scale strides.
        b_scl = tl.load(b_scl_block_ptr, mask=k_mask[:, None], other=1.0, cache_modifier=".ca")

        # --- Dequantize and Compute ---
        a_deq = (a_block.to(tl.float32) * a_scl.to(tl.float32)) * scale_a
        b_deq = (b_block.to(tl.float32) * b_scl.to(tl.float32)) * scale_b
        acc += tl.dot(a_deq, b_deq) # A@B.T logic

    # --- Store Output ---
    acc_bf16 = acc.to(tl.bfloat16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    m_mask = offs_m < M
    n_mask = offs_n < N
    out_mask = m_mask[:, None] & n_mask[None, :]

    if ATOMIC:
        tl.atomic_add(c_ptrs, acc_bf16, mask=out_mask)
    else:
        # This path shouldn't be hit if called from triton_mi300_fp8_mm with split_k > 1
        # but kept for potential direct kernel calls.
        tl.store(c_ptrs, acc_bf16, mask=out_mask)
# --- End of Triton Kernel ---


# --- Main Dispatcher Function ---
def triton_mi300_fp8_mm(
    a: torch.Tensor,       # Input FP8 tensor M x K
    b: torch.Tensor,       # Input FP8 tensor N x K
    a_scale: torch.Tensor, # Scale tensor for A
    b_scale: torch.Tensor, # Scale tensor for B
    scale_a: float = 1.0,
    scale_b: float = 1.0,
    split_k: int = 1,
):
    M, K = a.shape
    N, K_b = b.shape
    assert K == K_b, f"K dimension mismatch: A has {K}, B has {K_b}"
    assert a.dtype == torch.float8_e4m3fn or a.dtype == torch.float8_e5m2, "Input A must be FP8"
    assert b.dtype == torch.float8_e4m3fn or b.dtype == torch.float8_e5m2, "Input B must be FP8"
    # Add checks for scale dtypes and shapes if necessary depending on usage

    # Ensure inputs are contiguous (important for both paths)
    a = a.contiguous()
    b = b.contiguous() # B is N x K
    a_scale = a_scale.contiguous()
    b_scale = b_scale.contiguous()

    # Output tensor - BF16 for both paths
    output_dtype = torch.bfloat16

    if split_k == 1:
        # Use the user-provided custom_kernel
        # Allocate output tensor C for custom_kernel to fill
        c = torch.empty((M, N), device=a.device, dtype=output_dtype)
        # Package data for custom_kernel
        data: input_t = (a, b, a_scale, b_scale, c)
        c_output = custom_kernel(data)
        return c_output
    else:

        c = torch.zeros((M, N), device=a.device, dtype=output_dtype)

        # Triton grid calculation (2D grid)
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

        # Get strides for ALL tensors (data and scales)
        stride_am, stride_ak = a.stride()
        stride_bn, stride_bk = b.stride() # b is N x K
        stride_cm, stride_cn = c.stride()
        stride_a_scl_m, stride_a_scl_k = a_scale.stride() if a_scale.dim() == 2 else (0,0) # Crude handling, refine based on actual scale shapes
        stride_b_scl_n, stride_b_scl_k = b_scale.stride() if b_scale.dim() == 2 else (0,0) # Crude handling

        # Launch the kernel
        _kernel[grid](
            a_ptr=a, a_scl_ptr=a_scale,
            b_ptr=b, b_scl_ptr=b_scale, # Pass N x K tensors
            c_ptr=c,
            stride_am=stride_am, stride_ak=stride_ak,
            stride_bk=stride_bk, stride_bn=stride_bn, # Kernel expects KxN access, so use strides for NxK B
            stride_cm=stride_cm, stride_cn=stride_cn,
            # Pass scale strides
            stride_a_scl_m=stride_a_scl_m, stride_a_scl_k=stride_a_scl_k,
            stride_b_scl_n=stride_b_scl_n, stride_b_scl_k=stride_b_scl_k,
            scale_a=scale_a, scale_b=scale_b,
            M=M, N=N, K=K,
            ATOMIC=True, # Use atomic add for split_k > 1
            # BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages are handled by autotuner
        )
        return c

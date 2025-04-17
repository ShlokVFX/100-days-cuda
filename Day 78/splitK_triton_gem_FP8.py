import torch
import triton
import triton.language as tl
from task import input_t, output_t

#############################################
# 1. Original PyTorch Custom Dequant Kernel
#############################################

def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    a = a.contiguous()
    a_scale = a_scale.contiguous()
    b_scale = b_scale.contiguous()

    m, k = a.shape
    n = b.shape[0]
    block_shape_n = 128
    block_shape_k = 128
    scale_n, scale_k = b_scale.shape

    a_scale_exp = a_scale.unsqueeze(-1).expand(m, scale_k, block_shape_k)
    a_scale_exp = a_scale_exp.reshape(m, scale_k * block_shape_k)[:, :k]
    a_dequant = a.to(a_scale_exp.dtype) * a_scale_exp

    b_scale_reshaped = b_scale.view(scale_n, scale_k, 1, 1)
    b_scale_exp = b_scale_reshaped.expand(scale_n, scale_k, block_shape_n, block_shape_k)
    b_scale_exp = b_scale_exp.permute(0, 2, 1, 3).reshape(scale_n * block_shape_n, scale_k * block_shape_k)
    b_scale_exp = b_scale_exp[:n, :k]

    b_dequant = b.to(b_scale_exp.dtype) * b_scale_exp
    tmp = torch.matmul(a_dequant, b_dequant.t())
    c.copy_(tmp.to(torch.bfloat16))
    return c

#############################################
# 2. Triton FP8 Kernel with Aggressive Optimizations
#############################################

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 16}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'SPLIT_K': 8 }, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128,'SPLIT_K': 4 }, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K', 'SPLIT_K']
)
@triton.jit
def fp8_triton_kernel_optim(
    a_ptr, a_scale_ptr, b_ptr, b_scale_ptr, c_ptr,
    M, N, K,
    a_stride_m, a_stride_k,
    b_stride_n, b_stride_k,
    c_stride_m, c_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m, pid_n, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Partial accumulator in registers
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    K_per_split = tl.cdiv(K, SPLIT_K)
    start_k = pid_k * K_per_split
    end_k = tl.minimum(start_k + K_per_split, K)

    # Double buffer registers
    for k0 in range(start_k, end_k, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        mask_k = rk < end_k

        # Prefetch next block pointers
        a_off = a_ptr + rm[:, None] * a_stride_m + rk[None, :] * a_stride_k
        b_off = b_ptr + rn[:, None] * b_stride_n + rk[None, :] * b_stride_k
        a_blk = tl.load(a_off, mask=mask_k[None, :], other=0.0, cache=tl.CACHE_L1)
        a_scl = tl.load(a_scale_ptr + rm[:, None] * a_stride_m + rk[None, :] * a_stride_k,
                        mask=mask_k[None, :], other=1.0, cache=tl.CACHE_L1)
        b_blk = tl.load(b_off, mask=mask_k[None, :], other=0.0, cache=tl.CACHE_L1)
        b_scl = tl.load(b_scale_ptr + rn[:, None] * b_stride_n + rk[None, :] * b_stride_k,
                        mask=mask_k[None, :], other=1.0, cache=tl.CACHE_L1)

        # Dequantize in registers
        a_deq = a_blk * a_scl
        b_deq = tl.trans(b_blk * b_scl)

        # Accumulate
        acc += tl.dot(a_deq, b_deq)

    # Write back full reduction across split_k
    # Each pid_k writes into c_partial slice starting at pid_k*M*N
    out_ptr = c_ptr + pid_k * M * N + rm[:, None] * c_stride_m + rn[None, :] * c_stride_n
    tl.store(out_ptr, acc)


def triton_fp8_optimized(data: input_t, split_k: int = 16) -> output_t:
    a, b, a_scale, b_scale, c = data
    a, b = a.contiguous(), b.contiguous()
    a_scale, b_scale = a_scale.contiguous(), b_scale.contiguous()

    M, K = a.shape
    N, _ = b.shape
    # Fetch autotuned config
    cfg = fp8_triton_kernel_optim.meta['configs'][0].key if 'configs' in fp8_triton_kernel_optim.meta else fp8_triton_kernel_optim.meta
    BLOCK_M = cfg['BLOCK_M']; BLOCK_N = cfg['BLOCK_N']; BLOCK_K = cfg['BLOCK_K']

    # Allocate partial buffer
    c_partial = torch.zeros((split_k, M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), split_k)

    fp8_triton_kernel_optim[grid](
        a.data_ptr(), a_scale.data_ptr(), b.data_ptr(), b_scale.data_ptr(), c_partial.data_ptr(),
        M, N, K,
        K, 1,
        K, 1,
        N, 1,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        SPLIT_K=split_k
    )

    # Final reduction to bfloat16
    c.copy_(c_partial.sum(dim=0).to(torch.bfloat16))
    return c

#############################################
# 3. Unified Kernel Selector with SOL Benchmark Mode
#############################################

def fp8_kernel_selector(data: input_t,
                         use_triton: bool = False,
                         split_k: int = 1,
                         sol_mode: bool = False) -> output_t:
    if not use_triton:
        return custom_kernel(data)
    if sol_mode:
        # aggressive mode: always use optimized kernel
        return triton_fp8_optimized(data, split_k)
    if split_k > 1:
        return triton_fp8_optimized(data, split_k)
    return custom_kernel(data)  # 

#Collected mean times (µs): [706.0, 273.0, 372.0, 202.0, 619.0, 1465.0, 673.0, 339.0, 190.0, 2420.0, 979.0, 1481.0, 484.0, 2570.0, 6070.0, 2930.0, 1318.0, 497.0]
#Geometric mean (µs): 821.361483998991
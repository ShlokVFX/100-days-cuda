import os
import torch
import triton
import triton.language as tl
from task import input_t, output_t

@triton.jit
def dequant_matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Matrix strides
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    # Kernel meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    """Simple matrix multiplication: C = A @ B.T"""
    # Map program ID to the block of C being computed
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Program ID to block indices
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    
    # Block start indices
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # Create block arrays for C
    m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Bounds checking masks
    mask_m = m_offsets < M
    mask_n = n_offsets < N
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k_start in range(0, K, BLOCK_SIZE_K):
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE_K)
        mask_k = k_offsets < K
        
        # Load elements from A (M, K)
        a_ptrs = a_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load elements from B (N, K)
        b_ptrs = b_ptr + n_offsets[:, None] * stride_bn + k_offsets[None, :] * stride_bk
        b = tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Matrix multiplication for this block
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32).T)
    
    # Store output
    c_ptrs = c_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    
    # Make sure inputs are contiguous
    a = a.contiguous()
    b = b.contiguous()
    a_scale = a_scale.contiguous()
    b_scale = b_scale.contiguous()
    
    # Get matrix dimensions
    m, k = a.shape
    n = b.shape[0]
    block_shape_n, block_shape_k = 128, 128
    scale_n, scale_k = b_scale.shape
    
    # Dequantize matrices
    # Expand scales and dequantize
    a_scale_exp = a_scale.unsqueeze(-1).expand(m, scale_k, block_shape_k)
    a_scale_exp = a_scale_exp.reshape(m, scale_k * block_shape_k)[:, :k]
    a_deq = a.to(a_scale_exp.dtype) * a_scale_exp
    
    b_scale_r = b_scale.view(scale_n, scale_k, 1, 1)
    b_scale_exp = b_scale_r.expand(scale_n, scale_k, block_shape_n, block_shape_k)
    b_scale_exp = b_scale_exp.permute(0, 2, 1, 3).reshape(scale_n * block_shape_n, scale_k * block_shape_k)[:n, :k]
    b_deq = b.to(b_scale_exp.dtype) * b_scale_exp
    
    # For small matrices, just use torch.matmul
    if max(m, n, k) < 512:
        tmp = torch.matmul(a_deq, b_deq.t())
        c.copy_(tmp.to(torch.bfloat16))
        return c
    
    # For larger matrices, use Triton
    # Set block sizes for matrix multiplication
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    # Calculate grid dimensions properly as a tuple
    grid = (((m + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M) * ((n + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N),)
    
    # Launch kernel
    dequant_matmul_kernel[grid](
        a_deq, b_deq, c,
        m, n, k,
        a_deq.stride(0), a_deq.stride(1),
        b_deq.stride(0), b_deq.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    return c
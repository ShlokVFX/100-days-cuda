import os
import torch
import triton
import triton.language as tl
from task import input_t, output_t

# Original PyTorch reference kernel for FP8 dequant + GEMM

def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    a = a.contiguous(); a_scale = a_scale.contiguous(); b_scale = b_scale.contiguous()
    m, k = a.shape; n = b.shape[0]
    block_shape_n, block_shape_k = 128, 128
    scale_n, scale_k = b_scale.shape
    # expand scales and dequantize
    a_scale_exp = a_scale.unsqueeze(-1).expand(m, scale_k, block_shape_k)
    a_scale_exp = a_scale_exp.reshape(m, scale_k * block_shape_k)[:, :k]
    a_deq = a.to(a_scale_exp.dtype) * a_scale_exp
    b_scale_r = b_scale.view(scale_n, scale_k, 1, 1)
    b_scale_exp = b_scale_r.expand(scale_n, scale_k, block_shape_n, block_shape_k)
    b_scale_exp = b_scale_exp.permute(0,2,1,3).reshape(scale_n*block_shape_n, scale_k*block_shape_k)[:n,:k]
    b_deq = b.to(b_scale_exp.dtype) * b_scale_exp
    tmp = torch.matmul(a_deq, b_deq.t())
    c.copy_(tmp.to(torch.bfloat16)); return c

# Enable TMA for faster loads
os.environ['ENABLE_TMA']='1'

# Launch mapping utilities
@triton.jit
def column_major(pid, m, n, BM: tl.constexpr, BN: tl.constexpr):
    gm = tl.cdiv(m, BM)
    return pid % gm, pid // gm

# Autotuned GPU kernel writing to c_partial slices (no atomics)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'SPLIT_K': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64,  'SPLIT_K': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32,  'SPLIT_K': 1}, num_warps=4, num_stages=2),
    ],
    key=['M','N','K','SPLIT_K']
)
@triton.jit
def scaled_fp8_kernel_splitk(
    a_ptr, a_scl_ptr, b_ptr, b_scl_ptr, c_ptr,
    stride_am, stride_ak, stride_bk, stride_bn,
    scale_a: tl.constexpr, scale_b: tl.constexpr,
    M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    SPLIT_K: tl.constexpr
):
    pid_m, pid_n, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # tile offsets
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    # k-split bounds
    per = tl.cdiv(K, SPLIT_K)
    start = pid_k * per; end = tl.minimum(start + per, K)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    # loop over K blocks
    for k0 in range(start, end, BK):
        rk = k0 + tl.arange(0, BK)
        mask = rk < end
        # compute pointers
        a_off = a_ptr + (rm[:,None]*stride_am + rk[None,:]*stride_ak)
        b_off = b_ptr + (rk[:,None]*stride_bk + rn[None,:]*stride_bn)
        a_blk = tl.load(a_off, mask=mask[None,:], other=0.0, cache=tl.CACHE_CG)
        b_blk = tl.load(b_off, mask=mask[:,None], other=0.0, cache=tl.CACHE_CG)
        # load scales
        a_scl = tl.load(a_scl_ptr + (rm[:,None]*stride_am + rk[None,:]*stride_ak), mask=mask[None,:], other=1.0, cache=tl.CACHE_L1)
        b_scl = tl.load(b_scl_ptr + (rn[:,None]*stride_bn + rk[None,:]*stride_bk), mask=mask[:,None], other=1.0, cache=tl.CACHE_L1)
        # dequant + global scale
        a_deq = (a_blk.to(tl.float32) * a_scl) * scale_a
        b_deq = (b_blk.to(tl.float32) * b_scl) * scale_b
        acc += tl.dot(a_deq, tl.trans(b_deq))
    # write partial result
    offs = pid_k * M * N + (rm[:,None]*BN + rn[None,:])
    out_ptr = c_ptr + offs
    tl.store(out_ptr, acc)

# Python wrapper: allocate c_partial and reduce in Python

def triton_scaled_fp8_mm(
    a, b, a_scale, b_scale,
    scale_a: float=1.0, scale_b: float=1.0, split_k: int=4
):
    if split_k==1:
        return custom_kernel((a,b,a_scale,b_scale,torch.empty_like(a,dtype=torch.bfloat16)))
    M,K = a.shape; N = b.shape[0]
    # pick autotuned config
    cfg = scaled_fp8_kernel_splitk.meta['configs'][0].key
    BM, BN, BK = cfg['BLOCK_M'], cfg['BLOCK_N'], cfg['BLOCK_K']
    # prepare c_partial[split_k, M, N]
    c_partial = torch.zeros((split_k, M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN), split_k)
    # launch
    scaled_fp8_kernel_splitk[grid](
        a.data_ptr(), a_scale.data_ptr(), b.data_ptr(), b_scale.data_ptr(), c_partial.data_ptr(),
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        scale_a, scale_b, M, N, K, BM, BN, BK, split_k
    )
    # reduce and cast
    return c_partial.sum(dim=0).to(torch.float16)

#Collected mean times (µs): [707.0, 265.0, 369.0, 187.0, 734.0, 1561.0, 750.0, 340.0, 181.0, 2460.0, 974.0, 1509.0, 573.0, 2580.0, 6130.0, 3030.0, 1355.0, 485.0]
#Geometric mean (µs): 841.3297840869956
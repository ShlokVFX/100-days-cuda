import os
import torch
import triton
import triton.language as tl
from task import input_t, output_t

# PyTorch fallback (unchanged)
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

os.environ['ENABLE_TMA'] = '1'

# Highly unrolled, vectorized 2D FP8 dequant + BF16 GEMM
@triton.autotune(
    configs=[
        # wide tile, vector_size=8
        triton.Config({'BM': 512, 'BN': 256, 'BK': 64,  'SPLIT_K': 8},  num_warps=16, num_stages=4),
        triton.Config({'BM': 256, 'BN': 512, 'BK': 64,  'SPLIT_K': 8},  num_warps=16, num_stages=4),
        triton.Config({'BM': 512, 'BN': 512, 'BK': 32,  'SPLIT_K': 16}, num_warps=16, num_stages=5),
    ],
    key=['M','N','K','SPLIT_K']
)
@triton.jit
def mi300_fp8_mm_opt2(
    a_ptr, a_scl_ptr, b_ptr, b_scl_ptr, c_ptr,
    stride_am, stride_ak, stride_bk, stride_bn,
    scale_a: tl.constexpr, scale_b: tl.constexpr,
    M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT_K: tl.constexpr
):
    pid_m, pid_n, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    per_k = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * per_k
    k_end = tl.minimum(K, k_start + per_k)
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # fully unrolled K-loop with vectorized FP8 loads
    for k0 in tl.static_range(0, per_k, BK):
        base = k_start + k0
        rk = base + tl.arange(0, BK)
        mask = rk < k_end

        # vector load 8 FP8 scalars at once
        a_off = a_ptr + rm[:,None]*stride_am + rk[None,:]*stride_ak
        a_q = tl.load(a_off, mask=mask[None,:], other=0.0,
                      cache=tl.CACHE_CA, evict_hint=tl.EVICT_NONE, vector_size=8)
        a_s = tl.load(a_scl_ptr + rm[:,None]*stride_am + rk[None,:]*stride_ak,
                     mask=mask[None,:], other=1.0,
                     cache=tl.CACHE_L1, vector_size=8)
        b_off = b_ptr + rk[:,None]*stride_bk + rn[None,:]*stride_bn
        b_q = tl.load(b_off, mask=mask[:,None], other=0.0,
                      cache=tl.CACHE_CA, evict_hint=tl.EVICT_NONE, vector_size=8)
        b_s = tl.load(b_scl_ptr + rn[:,None]*stride_bn + rk[None,:]*stride_bk,
                     mask=mask[:,None], other=1.0,
                     cache=tl.CACHE_L1, vector_size=8)

        # dequantize and scale
        a_f = (a_q.to(tl.float32) * a_s) * scale_a
        b_f = (b_q.to(tl.float32) * b_s) * scale_b
        acc += tl.dot(a_f, tl.trans(b_f))

    # write back via atomic-add
    offs = rm[:,None] * N + rn[None,:]
    tl.atomic_add(c_ptr + offs, acc.to(tl.float16))


def triton_mi300_fp8_mm(
    a, b, a_scale, b_scale,
    scale_a: float = 1.0, scale_b: float = 1.0,
    split_k: int = 8,
):
    if split_k == 1:
        return custom_kernel((a, b, a_scale, b_scale,
                              torch.empty_like(a, dtype=torch.bfloat16)))
    M, K = a.shape; N = b.shape[0]
    # fetch autotuned tiles
    cfg = mi300_fp8_mm_opt2.meta['configs'][0].key
    BM, BN, BK = cfg['BM'], cfg['BN'], cfg['BK']
    c = torch.zeros((M, N), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN), split_k)
    mi300_fp8_mm_opt2[grid](
        a.data_ptr(), a_scale.data_ptr(),
        b.data_ptr(), b_scale.data_ptr(),
        c.data_ptr(),
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        scale_a, scale_b,
        M, N, K,
        BM=BM, BN=BN, BK=BK, SPLIT_K=split_k
    )
    return c

import os
import torch
import triton
import triton.language as tl
from task import input_t, output_t

# Original fallback for split_k == 1
# (unchanged custom dequant + matmul in PyTorch)
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
    b_scale_exp = b_scale_r.expand(scale_n, scale_k, block_shape_n, block_shape_k)
    b_scale_exp = (b_scale_exp.permute(0,2,1,3)
                   .reshape(scale_n*block_shape_n, scale_k*block_shape_k))[:n,:k]
    b_deq = b.to(b_scale_exp.dtype) * b_scale_exp
    tmp = torch.matmul(a_deq, b_deq.t())
    c.copy_(tmp.to(torch.bfloat16))
    return c

# Lightning-fast MI300 FP8 GEMM with split-K, autotune, and atomic-add
os.environ['ENABLE_TMA'] = '1'

@triton.autotune(
    configs=[
        triton.Config({'BM': 512, 'BN': 256, 'BK': 64,  'SPLIT_K': 8},  num_warps=16, num_stages=5),
        triton.Config({'BM': 256, 'BN': 256, 'BK': 128, 'SPLIT_K': 8},  num_warps=16, num_stages=5),
        triton.Config({'BM': 256, 'BN': 512, 'BK': 64,  'SPLIT_K': 16}, num_warps=16, num_stages=5),
    ],
    key=['M','N','K','SPLIT_K']
)
@triton.jit
def mi300_fp8_mm_fast(
    a_ptr, a_scl_ptr, b_ptr, b_scl_ptr, c_ptr,
    stride_am, stride_ak, stride_bk, stride_bn,
    scale_a: tl.constexpr, scale_b: tl.constexpr,
    M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # output tile offsets
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)

    # each k-slice
    per_k = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * per_k
    k_end = tl.minimum(K, k_start + per_k)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    # iterate over this slice in steps of BK
    for k0 in range(0, per_k, BK):
        k_base = k_start + k0
        rk = k_base + tl.arange(0, BK)
        mask_k = rk < k_end

        # load A and scales
        a_off = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
        a_blk = tl.load(a_off, mask=mask_k[None, :], other=0.0, cache=tl.CACHE_CA)
        a_scl = tl.load(
            a_scl_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=mask_k[None, :], other=1.0, cache=tl.CACHE_L1
        )

        # load B and scales
        b_off = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
        b_blk = tl.load(b_off, mask=mask_k[:, None], other=0.0, cache=tl.CACHE_CA)
        b_scl = tl.load(
            b_scl_ptr + (rn[:, None] * stride_bn + rk[None, :] * stride_bk),
            mask=mask_k[:, None], other=1.0, cache=tl.CACHE_L1
        )

        # dequant + global scale
        a_deq = (a_blk.to(tl.float32) * a_scl) * scale_a
        b_deq = (b_blk.to(tl.float32) * b_scl) * scale_b
        acc += tl.dot(a_deq, tl.trans(b_deq))

    # write back: atomic-add reduces all k-slices
    offs = rm[:, None] * N + rn[None, :]
    tl.atomic_add(c_ptr + offs, acc.to(tl.float16))


def triton_mi300_fp8_mm(
    a, b, a_scale, b_scale,
    scale_a: float = 1.0, scale_b: float = 1.0,
    split_k: int = 16,
):
    # fallback for split_k == 1
    if split_k == 1:
        return custom_kernel((a, b, a_scale, b_scale,
                              torch.empty_like(a, dtype=torch.bfloat16)))

    M, K = a.shape
    N = b.shape[0]
    BM, BN, BK = (
        mi300_fp8_mm_fast.meta['configs'][0].key['BM'],
        mi300_fp8_mm_fast.meta['configs'][0].key['BN'],
        mi300_fp8_mm_fast.meta['configs'][0].key['BK'],
    )

    # allocate & zero output
    c = torch.zeros((M, N), device=a.device, dtype=torch.float16)
    grid = (
        triton.cdiv(M, BM),
        triton.cdiv(N, BN),
        split_k
    )

    mi300_fp8_mm_fast[grid](
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
#Collected mean times (µs): [702.0, 264.0, 368.0, 188.0, 595.0, 1432.0, 663.0, 339.0, 226.0, 2490.0, 1054.0, 1579.0, 471.0, 2530.0, 6070.0, 2880.0, 1317.0, 476.0]
#Geometric mean (µs): 823.0388221731337
#Collected mean times (µs): [708.0, 276.0, 371.0, 206.0, 620.0, 1464.0, 676.0, 338.0, 194.0, 2420.0, 984.0, 1478.0, 493.0, 2550.0, 6010.0, 2910.0, 1320.0, 515.0]
#Geometric mean (µs): 825.3832773851976
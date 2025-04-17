import os
import torch
import triton
import triton.language as tl
from task import input_t, output_t

# Fallback for split_k == 1

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

@triton.autotune(
    configs=[
        triton.Config({'BM': 256, 'BN': 256, 'BK': 128, 'SPLIT_K': 8}, num_warps=16, num_stages=5),
        triton.Config({'BM': 256, 'BN': 512, 'BK': 64,  'SPLIT_K': 16}, num_warps=16, num_stages=5),
    ],
    key=['M','N','K','SPLIT_K']
)
@triton.jit
def mi300_dynamic_block_quant(
    a_ptr, b_ptr, c_ptr,
    stride_am, stride_ak, stride_bk, stride_bn,
    M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT_K: tl.constexpr
):
    pid_m, pid_n, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # tile coords
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)

    # K-slice bounds
    per_k = tl.cdiv(K, SPLIT_K)
    k0 = pid_k * per_k
    k1 = tl.minimum(K, k0 + per_k)

    # integer accumulator
    acc_i32 = tl.zeros((BM, BN), dtype=tl.int32)

    # dynamic block quant per k-chunk
    for ks in range(k0, k1, BK):
        ke = ks + BK
        rk = ks + tl.arange(0, BK)
        mask = rk < k1

        # load raw A, B in FP16
        a_off = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
        a_fp16 = tl.load(a_off, mask=mask[None, :], other=0.0)
        b_off = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
        b_fp16 = tl.load(b_off, mask=mask[:, None], other=0.0)

        # compute per-block max abs
        abs_a = tl.abs(a_fp16)
        abs_b = tl.abs(b_fp16)
        max_a = tl.maximum(abs_a, axis=1)  # shape [BM]
        max_b = tl.maximum(abs_b, axis=0)  # shape [BN]

        # quantization scale factors
        scale_a = max_a / 127.0 + 1e-6
        scale_b = max_b / 127.0 + 1e-6

        # quantize to int8
        a_q = tl.round(a_fp16 / scale_a[:, None])
        b_q = tl.round(b_fp16 / scale_b[None, :])
        a_i8 = tl.max(tl.min(a_q, 127), -128).to(tl.int8)
        b_i8 = tl.max(tl.min(b_q, 127), -128).to(tl.int8)

        # integer matmul
        a_i32 = a_i8.to(tl.int32)
        b_i32 = b_i8.to(tl.int32)
        acc_i32 += tl.dot(a_i32, tl.trans(b_i32))

    # dequantize final accum
    # outer product of scales: [BM,BN]
    scl_mat = scale_a[:, None] * scale_b[None, :]
    acc_fp32 = acc_i32.to(tl.float32) * scl_mat

    # write via atomic-add to avoid host reduction
    offs = rm[:, None] * N + rn[None, :]
    tl.atomic_add(c_ptr + offs, acc_fp32.to(tl.float16))


def triton_mi300_fp8_mm(
    a, b, a_scale, b_scale,
    scale_a: float = 1.0, scale_b: float = 1.0,
    split_k: int = 8,
):
    # fallback path
    if split_k == 1:
        return custom_kernel((a, b, a_scale, b_scale,
                              torch.empty_like(a, dtype=torch.bfloat16)))

    M, K = a.shape
    N = b.shape[0]

    # allocate output
    c = torch.zeros((M, N), device=a.device, dtype=torch.float16)

    # launch 3D grid
    BM = mi300_dynamic_block_quant.meta['configs'][0].key['BM']
    BN = mi300_dynamic_block_quant.meta['configs'][0].key['BN']
    BK = mi300_dynamic_block_quant.meta['configs'][0].key['BK']
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN), split_k)

    mi300_dynamic_block_quant[grid](
        a.data_ptr(), b.data_ptr(), c.data_ptr(),
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        M, N, K, BM=BM, BN=BN, BK=BK, SPLIT_K=split_k
    )
    return c

#Collected mean times (µs): [705.0, 275.0, 368.0, 204.0, 617.0, 1451.0, 676.0, 338.0, 193.0, 2390.0, 973.0, 1457.0, 497.0, 2520.0, 5920.0, 2880.0, 1297.0, 500.0]
#Geometric mean (µs): 818.1219153751781
import os
import torch
import triton
import triton.language as tl
from task import input_t, output_t

# Keep your original fallback for split_k == 1
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

# Enable TMA for faster loads if available
os.environ['ENABLE_TMA'] = '1'

@triton.jit
def mi300_fp8_mm_atomic(
    a_ptr, a_scl_ptr, b_ptr, b_scl_ptr, c_ptr,
    stride_am, stride_ak, stride_bk, stride_bn,
    scale_a: tl.constexpr, scale_b: tl.constexpr,
    M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    # program_ids over output tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # row/col indices for this tile
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # Unroll the K loop by static steps of BK
    for k0 in tl.static_range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        valid_k = rk < K

        # load A block and its scale, quantized to FP8
        a_off = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
        a_blk = tl.load(a_off, mask=valid_k[None, :], other=0.0, cache=tl.CACHE_CA)

        a_scl_off = a_scl_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
        a_scl = tl.load(a_scl_off, mask=valid_k[None, :], other=1.0, cache=tl.CACHE_L1)

        # load B block and its scale
        b_off = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
        b_blk = tl.load(b_off, mask=valid_k[:, None], other=0.0, cache=tl.CACHE_CA)

        b_scl_off = b_scl_ptr + (rn[:, None] * stride_bn + rk[None, :] * stride_bk)
        b_scl = tl.load(b_scl_off, mask=valid_k[:, None], other=1.0, cache=tl.CACHE_L1)

        # dequant + global scale
        a_deq = (a_blk.to(tl.float32) * a_scl) * scale_a
        b_deq = (b_blk.to(tl.float32) * b_scl) * scale_b

        # accumulate FP32
        acc += tl.dot(a_deq, tl.trans(b_deq))

    # final offset in C (flat)
    offs = rm[:, None] * N + rn[None, :]
    ptr_c = c_ptr + offs

    # atomic add into FP16 output
    out32 = acc
    out16 = out32.to(tl.float16)
    tl.atomic_add(ptr_c, out16)


def triton_mi300_fp8_mm(
    a, b, a_scale, b_scale,
    scale_a: float = 1.0, scale_b: float = 1.0,
    split_k: int = 1,
):
    # fallback to your PyTorch version for split_k=1
    if split_k == 1:
        return custom_kernel((a, b, a_scale, b_scale, 
                              torch.empty_like(a, dtype=torch.bfloat16)))

    # tile sizes tuned for MI300
    BM, BN, BK = 256, 128, 64
    M, K = a.shape
    N = b.shape[0]

    # zero‐out output
    c = torch.zeros((M, N), device=a.device, dtype=torch.float16)

    # launch 2D grid
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    mi300_fp8_mm_atomic[grid](
        a.data_ptr(), a_scale.data_ptr(),
        b.data_ptr(), b_scale.data_ptr(),
        c.data_ptr(),
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        scale_a, scale_b,
        M, N, K,
        BM=BM, BN=BN, BK=BK,
        # no SPLIT_K passed; reduction via atomics
    )
    return c

#Collected mean times (µs): [701.0, 261.0, 366.0, 188.0, 593.0, 1435.0, 649.0, 336.0, 177.0, 2340.0, 959.0, 1420.0, 459.0, 2500.0, 5930.0, 2890.0, 1262.0, 475.0]
#Geometric mean (µs): 793.4388491896277
import os
import torch
import triton
import triton.language as tl
from task import input_t, output_t

# ==== Custom kernel shim for benchmarking harness ====
# Exposes `custom_kernel` and routes to Triton implementation for split_k=1

def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    out = triton_mi300_fp8_mm(a, b, a_scale, b_scale, split_k=1)
    c.copy_(out)
    return c

# Enable TMA for faster loads if available
os.environ['ENABLE_TMA'] = '1'

@triton.jit
def mi300_fp8_mm(
    a_ptr, a_scl_ptr, b_ptr, b_scl_ptr, c_ptr,
    stride_am, stride_ak, stride_bk, stride_bn,
    scale_a: tl.constexpr, scale_b: tl.constexpr,
    M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    # Compute swizzled indices for FP8->FP32 promotion (no buffer used)
    lane = tl.arange(0, BM)
    swizzled_idx = (lane % 32) * 512 + (lane // 32)
    
    # program_ids over output tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # row/col indices for this tile
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k0 in tl.static_range(0, BK):
        rk = k0 + tl.arange(0, BK)
        valid_k = rk < K

        a_blk = tl.load(
            a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=valid_k[None, :], other=0.0
        )
        a_scl = tl.load(
            a_scl_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=valid_k[None, :], other=1.0
        )

        b_blk = tl.load(
            b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn),
            mask=valid_k[:, None], other=0.0
        )
        b_scl = tl.load(
            b_scl_ptr + (rn[:, None] * stride_bn + rk[None, :] * stride_bk),
            mask=valid_k[:, None], other=1.0
        )

        a_deq = (a_blk.to(tl.float32) * a_scl) * scale_a
        b_deq = (b_blk.to(tl.float32) * b_scl) * scale_b
        acc += tl.dot(a_deq, tl.trans(b_deq))

    offs = rm[:, None] * N + rn[None, :]
    ptr_c = c_ptr + offs
    tl.store(ptr_c, acc.to(tl.float16), mask=offs < (M * N))

@triton.jit
def mi300_fp8_mm_atomic(
    a_ptr, a_scl_ptr, b_ptr, b_scl_ptr, c_ptr,
    stride_am, stride_ak, stride_bk, stride_bn,
    scale_a: tl.constexpr, scale_b: tl.constexpr,
    M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    lane = tl.arange(0, BM)
    swizzled_idx = (lane % 32) * 512 + (lane // 32)

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k0 in tl.static_range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        valid_k = rk < K

        a_blk = tl.load(
            a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=valid_k[None, :], other=0.0
        )
        a_scl = tl.load(
            a_scl_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=valid_k[None, :], other=1.0
        )
        b_blk = tl.load(
            b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn),
            mask=valid_k[:, None], other=0.0
        )
        b_scl = tl.load(
            b_scl_ptr + (rn[:, None] * stride_bn + rk[None, :] * stride_bk),
            mask=valid_k[:, None], other=1.0
        )
        a_deq = (a_blk.to(tl.float32) * a_scl) * scale_a
        b_deq = (b_blk.to(tl.float32) * b_scl) * scale_b
        acc += tl.dot(a_deq, tl.trans(b_deq))

    offs = rm[:, None] * N + rn[None, :]
    out16 = acc.to(tl.float16)
    tl.atomic_add(c_ptr + offs, out16)


def triton_mi300_fp8_mm(
    a, b, a_scale, b_scale,
    scale_a: float = 1.0, scale_b: float = 1.0,
    split_k: int = 1,
) -> torch.Tensor:
    """
    Top-level Python entrypoint that dispatches to the appropriate Triton kernel.
    """
    BM, BN, BK = 256, 128, 64
    M, K = a.shape
    N = b.shape[0]
    c = torch.zeros((M, N), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    if split_k == 1:
        mi300_fp8_mm[grid](
            a.data_ptr(), a_scale.data_ptr(),
            b.data_ptr(), b_scale.data_ptr(),
            c.data_ptr(),
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            scale_a, scale_b,
            M, N, K,
            BM=BM, BN=BN, BK=BK,
        )
    else:
        mi300_fp8_mm_atomic[grid](
            a.data_ptr(), a_scale.data_ptr(),
            b.data_ptr(), b_scale.data_ptr(),
            c.data_ptr(),
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            scale_a, scale_b,
            M, N, K,
            BM=BM, BN=BN, BK=BK,
        )
    return c

# Collected mean times (µs): ...

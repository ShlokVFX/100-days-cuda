import os
import torch
import triton
import triton.language as tl
from task import input_t, output_t

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
    b_scale_exp = b_scale_exp.permute(0,2,1,3).reshape(scale_n*block_shape_n, scale_k*block_shape_k)[:n,:k]
    b_deq = b.to(b_scale_exp.dtype) * b_scale_exp

    tmp = torch.matmul(a_deq, b_deq.t())
    c.copy_(tmp.to(torch.bfloat16))
    return c


@triton.autotune(
    configs=[
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64}, num_stages=3, num_warps=4),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 64}, num_stages=3, num_warps=8),
        triton.Config({'BM': 128, 'BN': 64,  'BK': 64}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def mi300_fp8_mm_atomic(
    a_ptr, a_scl_ptr, b_ptr, b_scl_ptr, c_ptr,
    stride_am, stride_ak, stride_bk, stride_bn,
    scale_a: tl.constexpr, scale_b: tl.constexpr,
    M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k0 in tl.static_range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        valid_k = rk < K

        a_off = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_blk = tl.load(a_off, mask=valid_k[None, :], other=0.0)

        a_scl_off = a_scl_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_scl = tl.load(a_scl_off, mask=valid_k[None, :], other=1.0)

        b_off = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        b_blk = tl.load(b_off, mask=valid_k[:, None], other=0.0)

        b_scl_off = b_scl_ptr + rn[:, None] * stride_bn + rk[None, :] * stride_bk
        b_scl = tl.load(b_scl_off, mask=valid_k[:, None], other=1.0)

        a_deq = (a_blk.to(tl.float32) * a_scl) * scale_a
        b_deq = (b_blk.to(tl.float32) * b_scl) * scale_b

        acc += tl.dot(a_deq, tl.trans(b_deq))

    offs = rm[:, None] * N + rn[None, :]
    ptr_c = c_ptr + offs

    out16 = acc.to(tl.float16)
    tl.atomic_add(ptr_c, out16)


def triton_mi300_fp8_mm(
    a, b, a_scale, b_scale,
    scale_a: float = 1.0, scale_b: float = 1.0,
    split_k: int = 1,
):
    if split_k == 1:
        return custom_kernel((a, b, a_scale, b_scale,
                              torch.empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16, device=a.device)))

    M, K = a.shape
    N = b.shape[0]
    c = torch.zeros((M, N), dtype=torch.float16, device=a.device)

    grid = (triton.cdiv(M, 256), triton.cdiv(N, 128))
    mi300_fp8_mm_atomic[grid](
        a.data_ptr(), a_scale.data_ptr(),
        b.data_ptr(), b_scale.data_ptr(),
        c.data_ptr(),
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        scale_a, scale_b,
        M, N, K,
    )
    return c

#Collected mean times (µs): [705.0, 278.0, 369.0, 205.0, 622.0, 1474.0, 682.0, 339.0, 195.0, 2430.0, 989.0, 1473.0, 495.0, 2580.0, 6050.0, 2930.0, 1310.0, 505.0]
#Geometric mean (µs): 826.6443443747
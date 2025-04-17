#!POPCORN leaderboard amd-fp8-mm
#!POPCORN gpus MI300X

import torch
from task import input_t, output_t

# Enable high-performance matmul modes if supported
if torch.cuda.is_available() and hasattr(torch.backends.cuda, 'matmul'):
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    if hasattr(torch.backends.cuda.matmul, 'allow_bf16'):
        try:
            torch.backends.cuda.matmul.allow_bf16 = True
        except Exception:
            pass
    if hasattr(torch.backends.cuda.matmul, 'allow_fp8_matmul'):
        try:
            torch.backends.cuda.matmul.allow_fp8_matmul = True
        except Exception:
            pass

@torch.jit.script
def _dequant_fp16(a: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Fuse cast and multiply into one JITted kernel"""
    return a.to(torch.float16) * scale.to(torch.float16)

# PyTorch‑only kernel: JIT + TorchDynamo/Inductor
# Removed unsupported framework kwarg from torch.compile
@torch.compile(dynamic=False)
def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    # Ensure contiguous memory for maximal throughput
    a = a.contiguous()
    b = b.contiguous()
    a_scale = a_scale.contiguous()
    b_scale = b_scale.contiguous()

    # Dequantize to FP16 to leverage tensor cores
    a_fp16 = _dequant_fp16(a, a_scale)
    b_fp16 = _dequant_fp16(b, b_scale)

    # Fused GEMM: writes directly into preallocated BF16 output
    torch.matmul(a_fp16, b_fp16.transpose(0, 1), out=c)
    return c

# Optional Triton-based kernel for maximal MI300X utilization
import triton
import triton.language as tl

@triton.jit
def _triton_fp8_mm(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    a_scale_ptr, b_scale_ptr,
    stride_am, stride_ak, stride_bm, stride_bk, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Load and dequantize A block
    A = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    a_sc = tl.load(a_scale_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    A = A.to(tl.float16) * a_sc.to(tl.float16)

    # Load and dequantize B block
    B = tl.load(B_ptr + offs_n[:, None] * stride_bm + offs_k[None, :] * stride_bk)
    b_sc = tl.load(b_scale_ptr + offs_n[:, None] * stride_bm + offs_k[None, :] * stride_bk)
    B = B.to(tl.float16) * b_sc.to(tl.float16)

    # Compute blockwise dot
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc += tl.dot(A, B)
    acc = acc.to(tl.bfloat16)

    # Store result
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc)


def custom_kernel_triton(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    M, K = a.shape
    N = b.shape[0]
    # Tuned block sizes for MI300X
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
    _triton_fp8_mm[grid](
        a, b, c,
        M, N, K,
        a_scale, b_scale,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return c

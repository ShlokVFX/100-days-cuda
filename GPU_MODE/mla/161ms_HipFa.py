#!POPCORN leaderboard amd-mla-decode
#!POPCORN gpus MI300

import os
import sys
import math
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "clang++"
os.environ["PYTORCH_ROCM_ARCH"] = "gfx942"

cuda_src = r"""
#include <cmath>
#include <cstdint>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/amd_detail/amd_hip_runtime.h>

constexpr uint32_t cdiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

typedef __hip_bfloat16 bf16_t;

template <const uint32_t Bc, const uint32_t numThreads, const uint32_t dqk,
          const uint32_t dv>
__global__ void
decoding_flash_attention2_kernel(bf16_t *Q, bf16_t *K, bf16_t *V, bf16_t *O,
                                 const uint32_t B, const uint32_t nh,
                                 uint32_t N) {
  using uint16x2 =
      __attribute__((__vector_size__(2 * sizeof(uint16_t)))) uint16_t;
  using uint16x4 =
      __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
  using uint16x8 =
      __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
  using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

  static constexpr uint32_t VECTOR_SIZE_K = 8;
  static constexpr uint32_t VECTOR_SIZE_V = 4;
  static constexpr uint32_t WARPSIZE = 64;
  static constexpr uint32_t N_WARPS = numThreads / WARPSIZE;
  static constexpr uint32_t rowStrideV =
      (numThreads / (dv / VECTOR_SIZE_V)) * VECTOR_SIZE_V;

  static_assert(numThreads % (dv / VECTOR_SIZE_V) == 0,
                "numThreads should be divisible by dv / VECTOR_SIZE_V");
  static_assert(numThreads == Bc, "numThreads should be equal to Bc");

  uint64_t batchOffsetQ = (uint64_t)blockIdx.x * dqk;
  uint64_t batchOffsetK = (uint64_t)blockIdx.x * N * dqk;
  uint64_t batchOffsetV = (uint64_t)blockIdx.x * N * dv;
  uint64_t batchOffsetO = (uint64_t)blockIdx.x * dv;
  uint32_t tid = threadIdx.x;
  uint32_t colV = (tid % (dv / VECTOR_SIZE_V)) * VECTOR_SIZE_V;
  uint32_t rowV = (tid / (dv / VECTOR_SIZE_V)) * VECTOR_SIZE_V;

  uint32_t laneIdx = tid % WARPSIZE;
  uint32_t warpIdx = tid / WARPSIZE;

  __shared__ bf16_t Qs[dqk];
  __shared__ float Ss[Bc];
  __shared__ float Os[dv];

  float m = -INFINITY, l = 1.0f;

  Ss[tid] = 0.0f;
  // load data from global to smem
  for (uint32_t dqkIdx = tid; dqkIdx < dqk; dqkIdx += numThreads) {
    Qs[dqkIdx] = Q[batchOffsetQ + dqkIdx];
  }

  __syncthreads();

  for (uint32_t tileOffset = 0; tileOffset < N - 1; tileOffset += Bc) {
    for (uint32_t KsIdx = tid * VECTOR_SIZE_K; KsIdx < Bc * dqk;
         KsIdx += numThreads * VECTOR_SIZE_K) {
      uint32_t row = KsIdx / dqk;
      uint32_t col = KsIdx % dqk;

      uint16x8 qpack = *reinterpret_cast<uint16x8 *>(&Qs[col]);
      uint16x8 kpack = *reinterpret_cast<uint16x8 *>(
          &K[batchOffsetK + tileOffset * dqk + KsIdx]);

      float sum = 0.0f;
      for (uint32_t i = 0; i < VECTOR_SIZE_K; ++i) {
        uint16_t qu = qpack[i];
        uint16_t ku = kpack[i];
        float q = float(*reinterpret_cast<bf16_t *>(&qu));
        float k = float(*reinterpret_cast<bf16_t *>(&ku));
        sum += q * k;
      }
      atomicAdd(&Ss[row], sum); // TODO: use DPP
    }

    __syncthreads();

    float val = (float)bf16_t(Ss[tid]) / sqrtf(dqk);
    float m_local = val, l_local = 1.0f;

    for (uint32_t s = WARPSIZE / 2; s > 0; s /= 2) {
      float m_other = __shfl_down(m_local, s);
      float l_other = __shfl_down(l_local, s);
      if (m_other > m_local) {
        l_local *= expf(m_local - m_other);
        m_local = m_other;
      }
      l_local += l_other * expf(m_other - m_local);
    }

    m_local = __shfl(m_local, 0);
    l_local = __shfl(l_local, 0);

    float m_prev = m;

    if (m_local > m) {
      l *= expf(m - m_local);
      m = m_local;
    }
    l += l_local * expf(m_local - m);

    Ss[tid] = expf(val - m);

    for (uint32_t dvIdx = tid * 2; dvIdx < dv; dvIdx += numThreads * 2) {
      *reinterpret_cast<float2 *>(&Os[dvIdx]) =
          *reinterpret_cast<float2 *>(&Os[dvIdx]) * expf(m_prev - m);
    }

    __syncthreads();

    for (uint32_t rowOffsetV = 0; rowOffsetV < Bc; rowOffsetV += rowStrideV) {
      uint16x4 x[VECTOR_SIZE_V], xt[VECTOR_SIZE_V];
      floatx4 y;

      for (uint32_t i = 0; i < VECTOR_SIZE_V; ++i) {
        x[i] = *reinterpret_cast<uint16x4 *>(
            &V[batchOffsetV + (tileOffset + rowOffsetV + rowV + i) * dv +
               colV]);
      }

      for (uint32_t i = 0; i < VECTOR_SIZE_V; ++i) {
        for (uint32_t j = 0; j < VECTOR_SIZE_V; ++j) {
          xt[i][j] = x[j][i];
        }
      }

      y = *reinterpret_cast<floatx4 *>(&Ss[rowOffsetV + rowV]);

      for (uint32_t i = 0; i < VECTOR_SIZE_V; ++i) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < VECTOR_SIZE_V; ++j) {
          uint16_t au = xt[i][j];
          sum += float(*reinterpret_cast<bf16_t *>(&au)) * y[j];
        }
        atomicAdd(&Os[colV + i], sum);
      }
    }

    Ss[tid] = 0.0f;
    __syncthreads();
  }

  float sum = 0.0f;
  for (uint32_t dqkIdx = tid; dqkIdx < dqk; dqkIdx += numThreads) {
    sum += float(Qs[dqkIdx]) * float(K[batchOffsetK + (N - 1) * dqk + dqkIdx]);
  }
  for (uint32_t s = WARPSIZE / 2; s > 0; s /= 2) {
    sum += __shfl_down(sum, s);
  }
  sum = (float)bf16_t(__shfl(sum, 0)) / sqrtf(dqk);

  float m_prev = m;

  if (sum > m) {
    l *= expf(m - sum);
    m = sum;
  }
  l += expf(sum - m);

  for (uint32_t dvIdx = tid * 2; dvIdx < dv; dvIdx += numThreads * 2) {
    __hip_bfloat162 last = *reinterpret_cast<__hip_bfloat162 *>(
        &V[batchOffsetV + (N - 1) * dv + dvIdx]);

    float2 packf = *reinterpret_cast<float2 *>(&Os[dvIdx]);

    last.x = (packf.x * expf(m_prev - m) + (float)last.x * expf(sum - m)) / l;
    last.y = (packf.y * expf(m_prev - m) + (float)last.y * expf(sum - m)) / l;
    *reinterpret_cast<__hip_bfloat162 *>(&O[batchOffsetO + dvIdx]) = last;
  }
}

at::Tensor decoding_flash_attention2(at::Tensor Q, at::Tensor K, at::Tensor V) {
  assert(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous());
  assert(Q.sizes().size() == 4 && K.sizes().size() == 4 &&
         V.sizes().size() == 4 && Q.size(3) == 192 && K.size(3) == 192 &&
         V.size(3) == 128);

  uint32_t B = K.size(0), nh = K.size(1), N = K.size(2);

  const uint32_t dqk = 192;
  const uint32_t dv = 128;
  const uint32_t Bc = 64;

  at::Tensor O = at::empty({B, nh, 1, dv}, Q.options());

  const uint32_t numThreads = Bc;
  dim3 numBlocks(B * nh);
  decoding_flash_attention2_kernel<Bc, numThreads, dqk, dv>
      <<<numBlocks, numThreads>>>(
          (bf16_t *)Q.data_ptr(), (bf16_t *)K.data_ptr(),
          (bf16_t *)V.data_ptr(), (bf16_t *)O.data_ptr(), B, nh, N);

  return O;
}
"""

cpp_src = r"""
at::Tensor decoding_flash_attention2(at::Tensor Q, at::Tensor K, at::Tensor V);
"""

if sys.stdout is None:
    sys.stdout = open("/dev/stdout", "w")
if sys.stderr is None:
    sys.stderr = open("/dev/stderr", "w")

module = load_inline(
    name="decoding_flash_attention2",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["decoding_flash_attention2"],
    verbose=True,
    extra_cuda_cflags=[
        "-Ofast",
        "--offload-arch=gfx942",
        "-std=c++20",
        "-ffp-contract=fast",
        "-lhip_hcc",
         "-mcumode",
    ],
)

def precompute_basis(
    seq_len: int, head_dim: int, theta0: float = 10000
) -> tuple[torch.Tensor, torch.Tensor]:
    freq = theta0 ** (
        -torch.arange(0, head_dim, 2, dtype=torch.bfloat16, device="cuda") / head_dim
    )
    pos = torch.arange(seq_len, dtype=torch.bfloat16, device="cuda")
    theta = torch.outer(pos, freq)

    cos = torch.cos(theta).repeat((1, 2))
    sin = torch.sin(theta).repeat((1, 2))

    return cos, sin


def rope(
    x: torch.Tensor, basis: tuple[torch.Tensor, torch.Tensor], pos: int = 0
) -> torch.Tensor:
    cos, sin = basis
    x1, x2 = x.chunk(2, dim=-1)
    return (
        x * cos[pos : pos + x.shape[-2]]
        + torch.cat((-x2, x1), dim=-1) * sin[pos : pos + x.shape[-2]]
    )


rope = torch.compile(rope, mode="max-autotune")


import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=4),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    m_mask = offs_am < M
    n_mask = offs_bn < N
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offs = k * BLOCK_SIZE_K + offs_k
        k_mask = k_offs < K
        
        a_ptrs = x_ptr + offs_am[:, None] * stride_xm + k_offs[None, :] * stride_xk
        b_ptrs = w_ptr + k_offs[:, None] * stride_wk + offs_bn[None, :] * stride_wn
        
        a_mask = m_mask[:, None] & k_mask[None, :]
        b_mask = k_mask[:, None] & n_mask[None, :]
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        accumulator += tl.dot(a, b)

    c_ptrs = out_ptr + offs_am[:, None] * stride_om + offs_bn[None, :] * stride_on
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, accumulator.to(tl.bfloat16), mask=c_mask)


def triton_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    K_w, N = w.shape
    assert K == K_w
    
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    matmul_kernel[grid](
        x, w, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
    )
    
    return out


# REPLACE the class-based modules with these JIT functions:

@torch.jit.script
def q_projection_jit(
    x: torch.Tensor,
    q_proj_down_weight: torch.Tensor,
    q_proj_up_weight: torch.Tensor,
    cos_basis: torch.Tensor,
    sin_basis: torch.Tensor,
    prev_len: int
) -> torch.Tensor:
    
    nh = 128
    dq_nope = 128
    dq_rope = 64
    
    B, C, d = x.shape
    
    x_flat = x.view(-1, d)
    q_down = torch.mm(x_flat, q_proj_down_weight.t())
    q_up = torch.mm(q_down, q_proj_up_weight.t()).view(B, C, nh, dq_nope + dq_rope)
    q_up = q_up.transpose(1, 2)
    
    q_nope = q_up[..., :dq_nope]
    q_rope = q_up[..., dq_nope:]
    
    seq_len = q_rope.shape[-2]
    cos_slice = cos_basis[prev_len:prev_len + seq_len]
    sin_slice = sin_basis[prev_len:prev_len + seq_len]
    
    x1, x2 = q_rope.chunk(2, dim=-1)
    q_rope_rotated = q_rope * cos_slice + torch.cat((-x2, x1), dim=-1) * sin_slice
    
    q = torch.cat([q_nope, q_rope_rotated], dim=-1)
    return q.contiguous()

@torch.jit.script
def kv_projection_jit(
    x: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_proj_down_weight: torch.Tensor,
    kv_proj_up_weight: torch.Tensor,
    cos_basis: torch.Tensor,
    sin_basis: torch.Tensor,
    prev_len: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    nh = 128
    dkv = 512
    dk_nope = 128
    dk_rope = 64
    dv = 128
    
    B, C, d = x.shape
    total_len = prev_len + C
    
    x_flat = x.view(-1, d)
    kv_down = torch.mm(x_flat, kv_proj_down_weight.t()).view(B, C, -1)
    
    kv_cache[:, prev_len:total_len] = kv_down
    kv_full = kv_cache[:, :total_len]
    
    kv_down_part = kv_full[..., :dkv]
    k_rope_full = kv_full[..., dkv:]
    
    kv_up = torch.bmm(
        kv_down_part,
        kv_proj_up_weight.t().unsqueeze(0).expand(B, -1, -1)
    ).view(B, total_len, nh, dk_nope + dv).transpose(1, 2)
    
    k_nope = kv_up[..., :dk_nope]
    v = kv_up[..., dk_nope:]
    
    seq_len = k_rope_full.shape[-2]
    cos_slice = cos_basis[:seq_len]
    sin_slice = sin_basis[:seq_len]
    
    x1, x2 = k_rope_full.chunk(2, dim=-1)
    k_rope_rotated = k_rope_full * cos_slice + torch.cat((-x2, x1), dim=-1) * sin_slice
    k_rope_expanded = k_rope_rotated.unsqueeze(1).expand(-1, nh, -1, -1)
    k = torch.cat([k_nope, k_rope_expanded], dim=-1)
    
    return k.contiguous(), v.contiguous(), kv_cache

@torch.jit.script
def output_projection_jit(attn_output: torch.Tensor, wo_weight: torch.Tensor) -> torch.Tensor:
    nh = 128
    dv = 128
    B, C = attn_output.shape[0], attn_output.shape[2]
    y_flat = attn_output.transpose(1, 2).reshape(B, C, -1)
    out = torch.mm(y_flat.view(-1, nh * dv), wo_weight.t()).view(B, C, -1)
    return out

# REPLACE your compute function with this:
@torch.no_grad
def compute(
    x: torch.Tensor,
    kv_cache: torch.Tensor,
    prev_len: int,
    Q_proj_down_weight: torch.Tensor,
    Q_proj_up_weight: torch.Tensor,
    KV_proj_down_weight: torch.Tensor,
    KV_proj_up_weight: torch.Tensor,
    wo_weight: torch.Tensor,
):
    cos_basis, sin_basis = basis
    
    q = q_projection_jit(x, Q_proj_down_weight, Q_proj_up_weight, 
                         cos_basis, sin_basis, prev_len)
    
    k, v, updated_kv_cache = kv_projection_jit(x, kv_cache, KV_proj_down_weight, 
                                             KV_proj_up_weight, cos_basis, sin_basis, prev_len)
    
    y = module.decoding_flash_attention2(q, k, v)
    out = output_projection_jit(y, wo_weight)
    
    return out

# compute = torch.compile(compute, mode="max-autotune")

basis = precompute_basis(6145, 64)


def custom_kernel(data: tuple) -> tuple:
    config, x, kv_cache = data
    prev_len = kv_cache.seq_len
    out = compute(
        x,
        kv_cache.data,
        prev_len,
        config.Q_proj_down_weight,
        config.Q_proj_up_weight,
        config.KV_proj_down_weight,
        config.KV_proj_up_weight,
        config.wo_weight,
    )
    return out, kv_cache.data

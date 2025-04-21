#!POPCORN leaderboard amd-fp8-mm
#!POPCORN gpus MI300

import os
import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

# Use clang++ as the C++ compiler for ROCm compatibility
os.environ["CXX"] = "clang++"

# HIP kernel source for FP8 matrix multiplication using MFMA (Matrix Fused Multiply-Add)
cuda_src = r"""
#include <iostream>
#include <cstring>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

// Helper for ceiling division
__host__ __device__ __forceinline__ size_t cdiv(size_t a, size_t b) {
  return (a + b - 1) / b;
}

#define BLOCK_K 16
#define TILE_N 64
#define TILE_M 64
#define PAD 32  // pad shared memory to avoid bank conflicts
#pragma amdgcn register_pressure 96
#pragma amdgcn vgpr_spilling_off

static_assert(TILE_N % 4 == 0 && TILE_M % 4 == 0, "Tile dims must be multiples of vector size");

template <const size_t BN, const size_t BK, const size_t BM>
__global__ void fp8_mm_kernel(
    const __hip_fp8_e4m3_fnuz* __restrict__ A,
    const __hip_fp8_e4m3_fnuz* __restrict__ B,
    const float*              __restrict__ A_scale,
    const float*              __restrict__ B_scale,
    __hip_bfloat16*           __restrict__ C,
    size_t N, size_t K, size_t M) {

  using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
  constexpr size_t VECTOR_SIZE = 4;
  constexpr size_t WARPSIZE     = 64;
  constexpr size_t numThreads   = (BN * BM) / VECTOR_SIZE;

  size_t rowOffsetC = blockIdx.y * BN;
  size_t colOffsetC = blockIdx.x * BM;

  size_t innerColA = threadIdx.x % (BN / VECTOR_SIZE);
  size_t innerRowA = threadIdx.x / (BN / VECTOR_SIZE);
  size_t innerColB = threadIdx.x % (BM / VECTOR_SIZE);
  size_t innerRowB = threadIdx.x / (BM / VECTOR_SIZE);

  size_t laneIdx     = threadIdx.x % WARPSIZE;
  size_t warpIdx     = threadIdx.x / WARPSIZE;
  size_t warpColOff  = (warpIdx % (BM/32)) * 32;
  size_t warpRowOff  = (warpIdx/(BM/32)) * 32;
  size_t warpX       = laneIdx % 32;
  size_t warpY       = laneIdx / 32;

  // Shared memory for tiles and scales
  __shared__ __hip_fp8_e4m3_fnuz As[BK][BN + PAD];
  __shared__ __hip_fp8_e4m3_fnuz Bs[BK][BM + PAD];
  __shared__ float Ascl[BK][BN + PAD];
  __shared__ float Bscl[BK][BM + PAD];

  floatx16 acc = {0};

  for (size_t tile = 0; tile < K; tile += BK) {
    // Load A tile and scale
    for (size_t off = 0; off < BK; off += (numThreads / (BN / VECTOR_SIZE))) {
      size_t gi = tile + off + innerRowA;
      size_t gj = rowOffsetC + innerColA * VECTOR_SIZE;
      size_t si = (off + innerRowA) ^ ((off + innerRowA) >> 1);

      __hip_fp8x4_e4m3_fnuz tmpA = {0};
      float4      scaleA = make_float4(0);
      if (gi < K && gj < N) {
        tmpA = *reinterpret_cast<const __hip_fp8x4_e4m3_fnuz*>(&A[gi * N + gj]);
        scaleA = *reinterpret_cast<const float4*>(&A_scale[gi * N + gj]);
      }
      *reinterpret_cast<__hip_fp8x4_e4m3_fnuz*>(&As[si][innerColA * VECTOR_SIZE]) = tmpA;
      *reinterpret_cast<float4*>(&Ascl[si][innerColA * VECTOR_SIZE])          = scaleA;
    }

    // Load B tile and scale
    for (size_t off = 0; off < BK; off += (numThreads / (BM / VECTOR_SIZE))) {
      size_t gi = tile + off + innerRowB;
      size_t gj = colOffsetC + innerColB * VECTOR_SIZE;

      __hip_fp8x4_e4m3_fnuz tmpB = {0};
      float4      scaleB = make_float4(0);
      if (gi < K && gj < M) {
        tmpB = *reinterpret_cast<const __hip_fp8x4_e4m3_fnuz*>(&B[gi * M + gj]);
        scaleB = *reinterpret_cast<const float4*>(&B_scale[gi * M + gj]);
      }
      *reinterpret_cast<__hip_fp8x4_e4m3_fnuz*>(&Bs[off + innerRowB][innerColB * VECTOR_SIZE]) = tmpB;
      *reinterpret_cast<float4*>(&Bscl[off + innerRowB][innerColB * VECTOR_SIZE])             = scaleB;
    }

    __syncthreads();

    // MFMA compute
    for (size_t k2 = 0; k2 < BK; k2 += 16) {
      __hip_fp8_e4m3_fnuz a_frag[8], b_frag[8];
      for (int i = 0; i < 8; ++i) {
        a_frag[i] = As[k2 + warpY * 8 + i][warpRowOff + warpX];
        b_frag[i] = Bs[k2 + warpY * 8 + i][warpColOff + warpX];
      }
      floatx16 tmp = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
          *reinterpret_cast<long*>(a_frag), *reinterpret_cast<long*>(b_frag), (floatx16){0}, 0,0,0);

      // accumulate with per-vector scales
      for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
          float a_s = Ascl[k2 + warpY*8 + j*2 + i/2][warpRowOff + warpX];
          float b_s = Bscl[k2 + warpY*8 + j*2 + i/2][warpColOff + warpX];
          acc[i + j*4] += tmp[i + j*4] * a_s * b_s;
        }
      }
    }
    __syncthreads();
  }

  // Write out
  for (int j = 0; j < 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      size_t r  = rowOffsetC + warpRowOff + j*8 + (laneIdx/32)*4 + (laneIdx%32)/8;
      size_t cc = colOffsetC + warpColOff + (laneIdx%32)%8;
      if (r < N && cc < M) {
        C[r * M + cc] = (__hip_bfloat16)acc[i + j*4];
      }
    }
  }
}

// Host launcher
at::Tensor fp8_mm(
    at::Tensor A, at::Tensor B,
    at::Tensor A_scale, at::Tensor B_scale,
    at::Tensor C) {
  size_t N = A.size(0), K = A.size(1), M = B.size(1);
  constexpr size_t BK = BLOCK_K, BN = TILE_N, BM = TILE_M;
  dim3 nt((BN*BM)/16);
  dim3 nb(cdiv(M,BM), cdiv(N,BN));

  fp8_mm_kernel<BN,BK,BM><<<nb,nt>>>(
      reinterpret_cast<const __hip_fp8_e4m3_fnuz*>(A.data_ptr()),
      reinterpret_cast<const __hip_fp8_e4m3_fnuz*>(B.data_ptr()),
      A_scale.data_ptr<float>(),
      B_scale.data_ptr<float>(),
      reinterpret_cast<__hip_bfloat16*>(C.data_ptr()),
      N, K, M);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    printf("HIP kernel launch failed: %s\n", hipGetErrorString(err));
  }
  return C;
}
"""

# C++ API declaration
cpp_src = r"""
#include <torch/extension.h>
at::Tensor fp8_mm(at::Tensor A, at::Tensor B, at::Tensor A_scale, at::Tensor B_scale, at::Tensor C);
"""

# Compile extension
module = load_inline(
    name="fp8_mm", cpp_sources=[cpp_src], cuda_sources=[cuda_src],
    functions=["fp8_mm"], verbose=True,
    extra_cuda_cflags=["-O3","--offload-arch=gfx942","-std=c++20"],
)

def custom_kernel(data: input_t) -> output_t:
    a, b, a_s, b_s, c = data
    return module.fp8_mm(a, b, a_s, b_s, c)

# Geometric mean latency ref: ~362us

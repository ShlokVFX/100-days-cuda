#!POPCORN leaderboard amd-fp8-mm
#!POPCORN gpus MI300

import os
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
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/amd_detail/amd_hip_fp8.h>

// Helper for ceiling division
__host__ __device__ __forceinline__ size_t cdiv(size_t a, size_t b) {
  return (a + b - 1) / b;
}

#define BLOCK_DIM 128
#define PAD 1  // padding to avoid bank conflicts
// Add AMD register allocation hints for better register utilization
#pragma amdgcn register_pressure 96
#pragma amdgcn vgpr_spilling_off


// Main FP8 GEMM kernel with swizzled/padded shared memory
template <const size_t BN, const size_t BK, const size_t BM>
__global__ void fp8_mm_kernel(
    const __hip_fp8_e4m3_fnuz* __restrict__ A,
    const __hip_fp8_e4m3_fnuz* __restrict__ B,
    const float*              __restrict__ A_scale,
    const float*              __restrict__ B_scale,
    __hip_bfloat16*           __restrict__ C,
    size_t N, size_t K, size_t M) {

  using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
  static constexpr size_t VECTOR_SIZE = 4;
  static constexpr size_t WARPSIZE     = 64;
  static constexpr size_t numThreads   = (BN * BM) / 16;
  static constexpr size_t strideA      = (numThreads / (BN / VECTOR_SIZE));
  static constexpr size_t strideB      = (numThreads / (BM / VECTOR_SIZE));

  size_t rowOffsetC = blockIdx.y * BN;
  size_t colOffsetC = blockIdx.x * BM;
  size_t colOffsetA = rowOffsetC;
  size_t colOffsetB = colOffsetC;

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

  // Swizzled and padded shared memory
  __shared__ __hip_fp8_e4m3_fnuz As[BK][BN + PAD];
  __shared__ __hip_fp8_e4m3_fnuz Bs[BK][BM + PAD];
  __shared__ float Ws[BN + PAD];

  __hip_fp8_e4m3_fnuz a[8], b[8];
  floatx16 d = {0};

  for (size_t tile = 0; tile < K; tile += BK) {
    // Load A tile
    for (size_t off = 0; off < BK; off += strideA) {
      size_t gi = tile + off + innerRowA;
      size_t gj = colOffsetA + innerColA * VECTOR_SIZE;
      size_t si = (off + innerRowA) ^ ((off + innerRowA) >> 1);
      __hip_fp8x4_e4m3_fnuz tmp;
      if (gi < K && gj < N)
        tmp = *reinterpret_cast<const __hip_fp8x4_e4m3_fnuz*>(&A[gi * N + gj]);
      else {
        float4 z = make_float4(0,0,0,0);
        tmp = *reinterpret_cast<const __hip_fp8x4_e4m3_fnuz*>(&z);
      }
      *reinterpret_cast<__hip_fp8x4_e4m3_fnuz*>(&As[si][innerColA * VECTOR_SIZE]) = tmp;
      if (threadIdx.x < (BN / VECTOR_SIZE)) {
        float4 s = make_float4(0,0,0,0);
        if (gi < K && gj < N)
          s = *reinterpret_cast<const float4*>(&A_scale[(gi / BLOCK_DIM)*N + gj]);
        *reinterpret_cast<float4*>(&Ws[innerColA * VECTOR_SIZE]) = s;
      }
    }

    // Load B tile
    for (size_t off = 0; off < BK; off += strideB) {
      size_t gi = tile + off + innerRowB;
      size_t gj = colOffsetB + innerColB * VECTOR_SIZE;
      __hip_fp8x4_e4m3_fnuz tmp;
      if (gi < K && gj < M)
        tmp = *reinterpret_cast<const __hip_fp8x4_e4m3_fnuz*>(&B[gi * M + gj]);
      else {
        float4 z = make_float4(0,0,0,0);
        tmp = *reinterpret_cast<const __hip_fp8x4_e4m3_fnuz*>(&z);
      }
      *reinterpret_cast<__hip_fp8x4_e4m3_fnuz*>(&Bs[off + innerRowB][innerColB * VECTOR_SIZE]) = tmp;
      if (threadIdx.x == 0) {
        float s = 0;
        if (gi < K && gj < M)
          s = B_scale[(gi / BLOCK_DIM)*cdiv(M,BLOCK_DIM) + (gj / BLOCK_DIM)];
        Ws[BN] = s;
      }
    }

    __syncthreads();

    // MFMA compute
    for (size_t k2 = 0; k2 < BK; k2 += 16) {
      for (int i = 0; i < 8; ++i) {
        a[i] = As[k2 + warpY*8 + i][warpRowOff + warpX];
        b[i] = Bs[k2 + warpY*8 + i][warpColOff + warpX];
      }
      floatx16 c = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
          *reinterpret_cast<long*>(a), *reinterpret_cast<long*>(b), (floatx16){0}, 0,0,0);
      float bs = Ws[BN];
      for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i)
          d[i+j*4] += c[i+j*4] * Ws[warpRowOff + j*8 + warpY*4 + i] * bs;
    }
    __syncthreads();
  }

  // Write out
  for (int j = 0; j < 4; ++j)
    for (int i = 0; i < 4; ++i) {
      size_t r = rowOffsetC + warpRowOff + j*8 + warpY*4 + i;
      size_t cc= colOffsetC + warpColOff + warpX;
      if (r < N && cc < M)
        C[r*M + cc] = (__hip_bfloat16)d[i+j*4];
    }
}

// Host launcher
at::Tensor fp8_mm(
    at::Tensor A, at::Tensor B,
    at::Tensor A_scale, at::Tensor B_scale, at::Tensor C) {
  size_t N = A.size(0), K = A.size(1), M = B.size(0);
  const size_t BK=16, BN=64, BM=64;
  dim3 nt((BN*BM)/16), nb(cdiv(M,BM), cdiv(N,BN));
  fp8_mm_kernel<BN,BK,BM><<<nb,nt>>>(
      reinterpret_cast<const __hip_fp8_e4m3_fnuz*>(A.data_ptr()),
      reinterpret_cast<const __hip_fp8_e4m3_fnuz*>(B.data_ptr()),
      A_scale.data_ptr<float>(),
      B_scale.data_ptr<float>(),
      reinterpret_cast<__hip_bfloat16*>(C.data_ptr()),
      N,K,M);
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
    a,b,a_s,b_s,c = data
    return module.fp8_mm(a,b,a_s,b_s,c)

# Geometric mean latency ref: ~362us

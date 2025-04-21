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
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/amd_detail/amd_hip_fp8.h>

// Helper function for ceiling division
__host__ __device__ __forceinline__ size_t cdiv(size_t a, size_t b) {
  return (a + b - 1) / b;
}

// Block size constant used in tiling
#define BLOCK_DIM 128

// Main FP8 GEMM kernel using tiling, shared memory, and MFMA
// BN: rows per block, BK: depth per block, BM: cols per block
template <const size_t BN, const size_t BK, const size_t BM>
__global__ void fp8_mm_kernel(const __hip_fp8_e4m3_fnuz *A,
                              const __hip_fp8_e4m3_fnuz *B,
                              const float *A_scale, const float *B_scale,
                              __hip_bfloat16 *C, size_t N, size_t K, size_t M) {
  using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;  // SIMD-like float accumulator

  static constexpr size_t VECTOR_SIZE = 4;
  static constexpr size_t WARPSIZE = 64;
  static constexpr size_t numThreads = (BN * BM) / 16;
  static constexpr size_t strideA = (numThreads / (BN / VECTOR_SIZE));
  static constexpr size_t strideB = (numThreads / (BM / VECTOR_SIZE));

  // Matrix block offsets (row-major)
  size_t rowOffsetC = blockIdx.y * BN;
  size_t colOffsetC = blockIdx.x * BM;
  size_t colOffsetA = rowOffsetC;
  size_t colOffsetB = colOffsetC;

  // Per-thread indexing
  size_t innerColA = threadIdx.x % (BN / VECTOR_SIZE);
  size_t innerRowA = threadIdx.x / (BN / VECTOR_SIZE);
  size_t innerColB = threadIdx.x % (BM / VECTOR_SIZE);
  size_t innerRowB = threadIdx.x / (BM / VECTOR_SIZE);

  // Warp-level indexing
  size_t laneIdx = threadIdx.x % WARPSIZE;
  size_t warpIdx = threadIdx.x / WARPSIZE;
  size_t warpColOffset = (warpIdx % (BM / 32)) * 32;
  size_t warpRowOffset = (warpIdx / (BM / 32)) * 32;
  size_t warpX = laneIdx % 32;
  size_t warpY = laneIdx / 32;

  // Shared memory buffers for A, B tiles and scaling values
  __shared__ __hip_fp8_e4m3_fnuz As[BK][BN], Bs[BK][BM];
  __shared__ float Ws[BN];

  __hip_fp8_e4m3_fnuz a[8], b[8];
  floatx16 d = {0};  // Output accumulator

  // Loop over tiles of K dimension
  for (size_t tileOffset = 0; tileOffset < K; tileOffset += BK) {

    // Load matrix A tile to shared memory
    for (size_t innerRowOffsetA = 0; innerRowOffsetA < BK; innerRowOffsetA += strideA) {
      if ((tileOffset + innerRowOffsetA + innerRowA) < K &&
          (colOffsetA + innerColA * VECTOR_SIZE) < N) {
        __hip_fp8x4_e4m3_fnuz tmp = *(__hip_fp8x4_e4m3_fnuz *)&A[(tileOffset + innerRowOffsetA + innerRowA) * N + (colOffsetA + innerColA * VECTOR_SIZE)];
        *(__hip_fp8x4_e4m3_fnuz *)&As[innerRowOffsetA + innerRowA][innerColA * VECTOR_SIZE] = tmp;

        // Load scale for A
        if (threadIdx.x < (BN / VECTOR_SIZE)) {
          float4 tmp = *(float4 *)&A_scale[((tileOffset + innerRowOffsetA + innerRowA) / BLOCK_DIM) * N + (colOffsetA + innerColA * VECTOR_SIZE)];
          *(float4 *)&Ws[threadIdx.x * VECTOR_SIZE] = tmp;
        }
      } else {
        // Out-of-bounds padding with zeros
        *(__hip_fp8x4_e4m3_fnuz *)&As[innerRowOffsetA + innerRowA][innerColA * VECTOR_SIZE] = float4{0.0f, 0.0f, 0.0f, 0.0f};
      }
    }

    // Load matrix B tile to shared memory
    for (size_t innerRowOffsetB = 0; innerRowOffsetB < BK; innerRowOffsetB += strideB) {
      if ((tileOffset + innerRowOffsetB + innerRowB) < K &&
          (colOffsetB + innerColB) < M) {
        __hip_fp8x4_e4m3_fnuz tmp = *(__hip_fp8x4_e4m3_fnuz *)&B[(tileOffset + innerRowOffsetB + innerRowB) * M + (colOffsetB + innerColB * VECTOR_SIZE)];
        *(__hip_fp8x4_e4m3_fnuz *)&Bs[innerRowOffsetB + innerRowB][innerColB * VECTOR_SIZE] = tmp;

        if (threadIdx.x == BN) {
          Ws[BN] = B_scale[((tileOffset + innerRowOffsetB + innerRowB) / BLOCK_DIM) * cdiv(M, BLOCK_DIM) + ((colOffsetB + innerColB) / BLOCK_DIM)];
        }
      } else {
        *(__hip_fp8x4_e4m3_fnuz *)&Bs[innerRowOffsetB + innerRowB][innerColB * VECTOR_SIZE] = float4{0.0f, 0.0f, 0.0f, 0.0f};
      }
    }

    __syncthreads();

    // Perform multiply-accumulate using MFMA
    for (size_t BKOffset = 0; BKOffset < BK; BKOffset += 16) {
      for (size_t i = 0; i < 8; ++i) {
        a[i] = As[BKOffset + warpY * 8 + i][warpRowOffset + warpX];
        b[i] = Bs[BKOffset + warpY * 8 + i][warpColOffset + warpX];
      }

      floatx16 c = {0};
      c = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
          *reinterpret_cast<long *>(a), *reinterpret_cast<long *>(b), c, 0, 0, 0);

      // Apply scaling factors
      float b_scale = Ws[BN];
      for (size_t j = 0; j < 4; ++j) {
        for (size_t i = 0; i < 4; ++i) {
          d[i + j * 4] += c[i + j * 4] * Ws[warpRowOffset + j * 8 + warpY * 4 + i] * b_scale;
        }
      }
    }

    __syncthreads();
  }

  // Write final result to global memory
  for (size_t j = 0; j < 4; ++j) {
    for (size_t i = 0; i < 4; ++i) {
      if ((rowOffsetC + warpRowOffset + j * 8 + warpY * 4 + i) < N &&
          (colOffsetC + warpColOffset + warpX) < M) {
        C[(rowOffsetC + warpRowOffset + j * 8 + warpY * 4 + i) * M + (colOffsetC + warpColOffset + warpX)] = (__hip_bfloat16)d[i + j * 4];
      }
    }
  }
}

// Host-side launcher used by PyTorch
at::Tensor fp8_mm(at::Tensor A, at::Tensor B, at::Tensor A_scale,
                  at::Tensor B_scale, at::Tensor C) {
  size_t N = A.size(0), K = A.size(1), M = B.size(0);
  const size_t BK = 16, BN = 64, BM = 64;

  // Define launch configuration
  dim3 numThreads((BN * BM) / 16);
  dim3 numBlocks(cdiv(M, BM), cdiv(N, BN));

  // Launch kernel
  fp8_mm_kernel<BN, BK, BM><<<numBlocks, numThreads>>>(
      (__hip_fp8_e4m3_fnuz *)A.data_ptr(), (__hip_fp8_e4m3_fnuz *)B.data_ptr(),
      A_scale.data_ptr<float>(), B_scale.data_ptr<float>(),
      (__hip_bfloat16 *)C.data_ptr(), N, K, M);
  return C;
}
"""

# C++ API declaration for PyTorch extension
cpp_src = r"""
at::Tensor fp8_mm(at::Tensor A, at::Tensor B, at::Tensor A_scale,
                  at::Tensor B_scale, at::Tensor C);
"""

# Compile the CUDA/HIP extension using torch's inline loader
module = load_inline(
    name="fp8_mm",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["fp8_mm"],
    verbose=True,
    extra_cuda_cflags=["-O3", "--offload-arch=gfx942", "-std=c++20"],
)

# PyTorch interface to call the HIP kernel
def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    return module.fp8_mm(a, b, a_scale, b_scale, c)

# Geometric mean latency reference: ~362 microseconds

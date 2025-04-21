#!POPCORN leaderboard amd-fp8-mm
#!POPCORN gpus MI300

import os
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "clang++"

cuda_src = r"""
#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/amd_detail/amd_hip_fp8.h>

__host__ __device__ __forceinline__ size_t cdiv(size_t a, size_t b) {
  return (a + b - 1) / b;
}

#define BLOCK_DIM 128

template <const size_t BN, const size_t BK, const size_t BM>
__global__ void fp8_mm_kernel(const __hip_fp8_e4m3_fnuz *A,
                              const __hip_fp8_e4m3_fnuz *B,
                              const float *A_scale, const float *B_scale,
                              __hip_bfloat16 *C, size_t N, size_t K, size_t M) {
  using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

  static constexpr size_t VECTOR_SIZE = 4;
  static constexpr size_t WARPSIZE = 64;
  static constexpr size_t numThreads = (BN * BM) / 16;
  static constexpr size_t strideA = (numThreads / (BN / VECTOR_SIZE));
  static constexpr size_t strideB = (numThreads / (BM / VECTOR_SIZE));

  size_t rowOffsetC = blockIdx.y * BN;
  size_t colOffsetC = blockIdx.x * BM;
  size_t colOffsetA = rowOffsetC;
  size_t colOffsetB = colOffsetC;

  static_assert(numThreads % BN == 0, "BN should be a multiple of numThreads");
  static_assert(numThreads % BM == 0, "BM should be a multiple of numThreads");
  static_assert(BN <= 128 && BK <= 128 && BM <= 128,
                "Range above 128 is not supported");

  size_t innerColA = threadIdx.x % (BN / VECTOR_SIZE);
  size_t innerRowA = threadIdx.x / (BN / VECTOR_SIZE);
  size_t innerColB = threadIdx.x % (BM / VECTOR_SIZE);
  size_t innerRowB = threadIdx.x / (BM / VECTOR_SIZE);

  size_t laneIdx = threadIdx.x % WARPSIZE;
  size_t warpIdx = threadIdx.x / WARPSIZE;
  size_t warpColOffset = (warpIdx % (BM / 32)) * 32;
  size_t warpRowOffset = (warpIdx / (BM / 32)) * 32;
  size_t warpX = laneIdx % 32;
  size_t warpY = laneIdx / 32;

  __shared__ __hip_fp8_e4m3_fnuz As[BK][BN], Bs[BK][BM];
  __shared__ float Ws[BN + 1];  // Add +1 to handle BN index

  __hip_fp8_e4m3_fnuz a[8], b[8];
  floatx16 d = {0};

  for (size_t tileOffset = 0; tileOffset < K; tileOffset += BK) {
    for (size_t innerRowOffsetA = 0; innerRowOffsetA < BK;
         innerRowOffsetA += strideA) {
      if ((tileOffset + innerRowOffsetA + innerRowA) < K &&
          (colOffsetA + innerColA * VECTOR_SIZE) < N &&
          (innerRowOffsetA + innerRowA) < BK &&
          (innerColA * VECTOR_SIZE) < BN) {
        __hip_fp8x4_e4m3_fnuz tmp =
            *(__hip_fp8x4_e4m3_fnuz
                  *)&A[(tileOffset + innerRowOffsetA + innerRowA) * N +
                       (colOffsetA + innerColA * VECTOR_SIZE)];
        *(__hip_fp8x4_e4m3_fnuz *)&As[innerRowOffsetA + innerRowA]
                                     [innerColA * VECTOR_SIZE] = tmp;
        if (threadIdx.x < (BN / VECTOR_SIZE)) {
          float4 tmp =
              *(float4 *)&A_scale[((tileOffset + innerRowOffsetA + innerRowA) /
                                   BLOCK_DIM) *
                                      N +
                                  (colOffsetA + innerColA * VECTOR_SIZE)];
          *(float4 *)&Ws[threadIdx.x * VECTOR_SIZE] = tmp;
        }
      } else if ((innerRowOffsetA + innerRowA) < BK &&
                 (innerColA * VECTOR_SIZE) < BN) {
        *(__hip_fp8x4_e4m3_fnuz *)&As[innerRowOffsetA + innerRowA]
                                     [innerColA * VECTOR_SIZE] =
            float4{0.0f, 0.0f, 0.0f, 0.0f};
      }
    }
    for (size_t innerRowOffsetB = 0; innerRowOffsetB < BK;
         innerRowOffsetB += strideB) {
      if ((tileOffset + innerRowOffsetB + innerRowB) < K &&
          (colOffsetB + innerColB) < M && (innerRowOffsetB + innerRowB) < BK &&
          (innerColB * VECTOR_SIZE) < BM) {
        __hip_fp8x4_e4m3_fnuz tmp =
            *(__hip_fp8x4_e4m3_fnuz
                  *)&B[(tileOffset + innerRowOffsetB + innerRowB) * M +
                       (colOffsetB + innerColB * VECTOR_SIZE)];
        *(__hip_fp8x4_e4m3_fnuz *)&Bs[innerRowOffsetB + innerRowB]
                                     [innerColB * VECTOR_SIZE] = tmp;
        if (threadIdx.x == BN) {
          Ws[BN] =
              B_scale[((tileOffset + innerRowOffsetB + innerRowB) / BLOCK_DIM) *
                          cdiv(M, BLOCK_DIM) +
                      ((colOffsetB + innerColB) / BLOCK_DIM)];
        }
      } else if ((innerRowOffsetB + innerRowB) < BK &&
                 (innerColB * VECTOR_SIZE) < BM) {
        *(__hip_fp8x4_e4m3_fnuz *)&Bs[innerRowOffsetB + innerRowB]
                                     [innerColB * VECTOR_SIZE] =
            float4{0.0f, 0.0f, 0.0f, 0.0f};
      }
    }

    __syncthreads();

    for (size_t BKOffset = 0; BKOffset < BK; BKOffset += 16) {
      
      for (size_t i = 0; i < 8; ++i) {
        a[i] = As[BKOffset + warpY * 8 + i][warpRowOffset + warpX];
        b[i] = Bs[BKOffset + warpY * 8 + i][warpColOffset + warpX];
      }
      
      // Option 1: Using MFMA instruction (current implementation)
      floatx16 c = {0};
      c = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
          *reinterpret_cast<long *>(a), *reinterpret_cast<long *>(b), c, 0, 1,
          0);
      
      // Alternative approach: Performing explicit conversion and computation
      // This shows how to use the intrinsic conversion for manual implementation
      // Uncomment this block to use explicit FP8 conversion instead of MFMA
      /*
      floatx16 c_alt = {0};
      
      // For demonstration purposes - a manual implementation with FP8->FP32 conversion
      // Note: In practice, the MFMA instruction above is more efficient
      for (size_t j = 0; j < 4; ++j) {
        for (size_t i = 0; i < 4; ++i) {
          float sum = 0.0f;
          for (size_t k = 0; k < 8; ++k) {
            // Convert FP8 to FP32 using intrinsics
            float a_val = __builtin_amdgcn_cvt_f32_fp8(
                As[BKOffset + k][warpRowOffset + j * 8 + i], 0);
            float b_val = __builtin_amdgcn_cvt_f32_fp8(
                Bs[BKOffset + k][warpColOffset + i * 8 + j], 0);
            sum += a_val * b_val;
          }
          c_alt[i + j * 4] = sum;
        }
      }
      */
      
      // Scale the results with the quantization factors
      float b_scale = Ws[BN];
      for (size_t j = 0; j < 4; ++j) {
        for (size_t i = 0; i < 4; ++i) {
          d[i + j * 4] += c[i + j * 4] *
                          Ws[warpRowOffset + j * 8 + warpY * 4 + i] * b_scale;
        }
      }
    }

    __syncthreads();
  }

  for (size_t j = 0; j < 4; ++j) {
    for (size_t i = 0; i < 4; ++i) {
      if ((rowOffsetC + warpRowOffset + j * 8 + warpY * 4 + i) < N &&
          (colOffsetC + warpColOffset + warpX) < M)
        C[(rowOffsetC + warpRowOffset + j * 8 + warpY * 4 + i) * M +
          (colOffsetC + warpColOffset + warpX)] = (__hip_bfloat16)d[i + j * 4];
    }
  }
}

// Add a second version of the kernel that uses explicit FP8 conversions
template <const size_t BN, const size_t BK, const size_t BM>
__global__ void fp8_mm_kernel_explicit_conversion(
    const __hip_fp8_e4m3_fnuz *A, const __hip_fp8_e4m3_fnuz *B,
    const float *A_scale, const float *B_scale, __hip_bfloat16 *C,
    size_t N, size_t K, size_t M) {
  using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

  static constexpr size_t VECTOR_SIZE = 4;
  static constexpr size_t WARPSIZE = 64;
  static constexpr size_t numThreads = (BN * BM) / 16;
  
  // Thread/block position calculations for matrix access
  size_t rowOffsetC = blockIdx.y * BN;
  size_t colOffsetC = blockIdx.x * BM;
  
  size_t laneIdx = threadIdx.x % WARPSIZE;
  size_t warpIdx = threadIdx.x / WARPSIZE;
  size_t warpRow = (warpIdx / (BM / 32)) * 16 + (laneIdx / 16) * 8;
  size_t warpCol = (warpIdx % (BM / 32)) * 32 + (laneIdx % 16) * 2;
  
  // Handle boundary conditions
  if (rowOffsetC + warpRow >= N || colOffsetC + warpCol >= M) return;
  
  // Accumulator for results
  float acc[4] = {0.0f};
  
  for (size_t k = 0; k < K; k++) {
    // Load values and convert FP8 -> FP32
    float a_values[2], b_values[2];
    
    // Load A matrix elements and convert using intrinsics
    for (int i = 0; i < 2; i++) {
      if (rowOffsetC + warpRow + i < N && k < K) {
        a_values[i] = __builtin_amdgcn_cvt_f32_fp8(
            A[(rowOffsetC + warpRow + i) * K + k], 0);
        // Apply quantization scale
        a_values[i] *= A_scale[((rowOffsetC + warpRow + i) / BLOCK_DIM) * 
                               ((k) / BLOCK_DIM)];
      } else {
        a_values[i] = 0.0f;
      }
    }
    
    // Load B matrix elements and convert using intrinsics
    for (int i = 0; i < 2; i++) {
      if (k < K && colOffsetC + warpCol + i < M) {
        b_values[i] = __builtin_amdgcn_cvt_f32_fp8(
            B[k * M + (colOffsetC + warpCol + i)], 0);
        // Apply quantization scale
        b_values[i] *= B_scale[(k / BLOCK_DIM) * 
                              ((colOffsetC + warpCol + i) / BLOCK_DIM)];
      } else {
        b_values[i] = 0.0f;
      }
    }
    
    // Perform matrix multiplication using explicit conversions
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        acc[i*2 + j] += a_values[i] * b_values[j];
      }
    }
  }
  
  // Write results back to C matrix
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      if (rowOffsetC + warpRow + i < N && colOffsetC + warpCol + j < M) {
        C[(rowOffsetC + warpRow + i) * M + (colOffsetC + warpCol + j)] = 
            (__hip_bfloat16)acc[i*2 + j];
      }
    }
  }
}

at::Tensor fp8_mm(at::Tensor A, at::Tensor B, at::Tensor A_scale,
                  at::Tensor B_scale, at::Tensor C) {
  size_t N = A.size(0), K = A.size(1), M = B.size(0);

  const size_t BK = 16;
  const size_t BN = 64;
  const size_t BM = 64;
  
  // Use the original optimized kernel that uses MFMA instructions
  dim3 numThreads((BN * BM) / 16);
  dim3 numBlocks(cdiv(M, BM), cdiv(N, BN));
  
  fp8_mm_kernel<BN, BK, BM><<<numBlocks, numThreads>>>(
      (__hip_fp8_e4m3_fnuz *)A.data_ptr(), (__hip_fp8_e4m3_fnuz *)B.data_ptr(),
      A_scale.data_ptr<float>(), B_scale.data_ptr<float>(),
      (__hip_bfloat16 *)C.data_ptr(), N, K, M);
      
  /* Uncomment to use the explicit conversion kernel
  // Alternatively, use the kernel with explicit FP8->FP32 conversions
  // Note: This is likely slower than the MFMA version
  dim3 numThreads2(256);
  dim3 numBlocks2(cdiv(M, 32), cdiv(N, 16));
  fp8_mm_kernel_explicit_conversion<BN, BK, BM><<<numBlocks2, numThreads2>>>(
      (__hip_fp8_e4m3_fnuz *)A.data_ptr(), (__hip_fp8_e4m3_fnuz *)B.data_ptr(),
      A_scale.data_ptr<float>(), B_scale.data_ptr<float>(),
      (__hip_bfloat16 *)C.data_ptr(), N, K, M);
  */
      
  return C;
}
"""

cpp_src = r"""
at::Tensor fp8_mm(at::Tensor A, at::Tensor B, at::Tensor A_scale,
                  at::Tensor B_scale, at::Tensor C);
"""

module = load_inline(
    name="fp8_mm",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["fp8_mm"],
    verbose=True,
    extra_cuda_cflags=[
        "-O3", 
        "--offload-arch=gfx942", 
        "-std=c++20",
    ],
)


def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    return module.fp8_mm(a, b, a_scale, b_scale, c)
#!POPCORN leaderboard amd-fp8-mm
#!POPCORN gpus MI300

import os
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "clang++"

cuda_src = r"""
#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__host__ __device__ __forceinline__ size_t cdiv(size_t a, size_t b) {
  return (a + b - 1) / b;
}

#define BLOCK_DIM 128

#define VECTOR_SIZE 4
#define WARPSIZE 64

template <const size_t BN, const size_t BK, const size_t BM>
__global__ void fp16_mm_kernel(const __half *A,
                                const __half *B,
                                __half *C,
                                size_t N, size_t K, size_t M) {
  static constexpr size_t numThreads = (BN * BM) / 16;
  static constexpr size_t strideA = (numThreads / (BN / VECTOR_SIZE));
  static constexpr size_t strideB = (numThreads / (BM / VECTOR_SIZE));

  size_t rowOffsetC = blockIdx.y * BN;
  size_t colOffsetC = blockIdx.x * BM;
  size_t colOffsetA = rowOffsetC;
  size_t colOffsetB = colOffsetC;

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

  __shared__ __half As[2][BK][BN], Bs[2][BK][BM];

  __half a[8], b[8];
  float d[16] = {0};

  int current_buffer = 0;
  int next_buffer = 1;

  size_t tileOffset = 0;

  __syncthreads();

  for (tileOffset = 0; tileOffset < K; tileOffset += BK) {
    for (size_t innerRowOffsetA = 0; innerRowOffsetA < BK; innerRowOffsetA += strideA) {
      if ((tileOffset + innerRowOffsetA + innerRowA) < K &&
          (colOffsetA + innerColA * VECTOR_SIZE) < N) {
        for (int i = 0; i < VECTOR_SIZE; i++) {
          As[current_buffer][innerRowOffsetA + innerRowA][innerColA * VECTOR_SIZE + i] =
              A[(tileOffset + innerRowOffsetA + innerRowA) * N + (colOffsetA + innerColA * VECTOR_SIZE + i)];
        }
      } else {
        for (int i = 0; i < VECTOR_SIZE; i++) {
          As[current_buffer][innerRowOffsetA + innerRowA][innerColA * VECTOR_SIZE + i] = __float2half(0.0f);
        }
      }
    }

    for (size_t innerRowOffsetB = 0; innerRowOffsetB < BK; innerRowOffsetB += strideB) {
      if ((tileOffset + innerRowOffsetB + innerRowB) < K &&
          (colOffsetB + innerColB * VECTOR_SIZE) < M) {
        for (int i = 0; i < VECTOR_SIZE; i++) {
          Bs[current_buffer][innerRowOffsetB + innerRowB][innerColB * VECTOR_SIZE + i] =
              B[(tileOffset + innerRowOffsetB + innerRowB) * M + (colOffsetB + innerColB * VECTOR_SIZE + i)];
        }
      } else {
        for (int i = 0; i < VECTOR_SIZE; i++) {
          Bs[current_buffer][innerRowOffsetB + innerRowB][innerColB * VECTOR_SIZE + i] = __float2half(0.0f);
        }
      }
    }

    __syncthreads();

    for (size_t BKOffset = 0; BKOffset < BK; BKOffset += 16) {
      for (size_t i = 0; i < 8; ++i) {
        a[i] = As[current_buffer][BKOffset + warpY * 8 + i][warpRowOffset + warpX];
        b[i] = Bs[current_buffer][BKOffset + warpY * 8 + i][warpColOffset + warpX];
      }

      float c[16] = {0};
      long4 a_val = *reinterpret_cast<long4 *>(a);
      long4 b_val = *reinterpret_cast<long4 *>(b);

      float* c_ptr = reinterpret_cast<float *>(c);
      float* d_ptr = reinterpret_cast<float *>(d);

      float result[16] = {0};
      auto mfma_out = __builtin_amdgcn_mfma_f32_32x32x16_f16_f16(a_val, b_val, *c_ptr, 0, 0, 0);

      for (int i = 0; i < 16; i++) d_ptr[i] += mfma_out;
    }

    __syncthreads();
    current_buffer = 1 - current_buffer;
    next_buffer = 1 - next_buffer;
  }

  for (size_t j = 0; j < 4; ++j) {
    for (size_t i = 0; i < 4; ++i) {
      if ((rowOffsetC + warpRowOffset + j * 8 + warpY * 4 + i) < N &&
          (colOffsetC + warpColOffset + warpX) < M) {
        C[(rowOffsetC + warpRowOffset + j * 8 + warpY * 4 + i) * M + (colOffsetC + warpColOffset + warpX)] = __float2half(d[i + j * 4]);
      }
    }
  }
}

at::Tensor fp8_mm(at::Tensor A, at::Tensor B, at::Tensor A_scale,
                  at::Tensor B_scale, at::Tensor C) {
  size_t N = A.size(0), K = A.size(1), M = B.size(0);
  const size_t BK = 32, BN = 128, BM = 128;
  dim3 numThreads((BN * BM) / 16);
  dim3 numBlocks(cdiv(M, BM), cdiv(N, BN));
  fp8_mm_kernel<BN, BK, BM><<<numBlocks, numThreads>>>(
      (__hip_fp8_e4m3_fnuz *)A.data_ptr(), (__hip_fp8_e4m3_fnuz *)B.data_ptr(),
      A_scale.data_ptr<float>(), B_scale.data_ptr<float>(),
      (__hip_bfloat16 *)C.data_ptr(), N, K, M);
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
    extra_cuda_cflags=["-O3", "--offload-arch=gfx942", "-std=c++20", "-ffp-contract=fast"],
)

def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    return module.fp8_mm(a, b, a_scale, b_scale, c)

#Collected mean times (µs): [418.0, 99.1, 420.0, 51.0, 255.0, 468.0, 286.0, 394.0, 45.9, 946.0, 413.0, 451.0, 183.0, 1236.0, 2800.0, 1386.0, 420.0, 212.0]
#Geometric mean (µs): 353.7947122881542
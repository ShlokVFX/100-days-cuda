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
  __shared__ float Ws[BN];

  __hip_fp8_e4m3_fnuz a[8], b[8];
  floatx16 d = {0};

  for (size_t tileOffset = 0; tileOffset < K; tileOffset += BK) {
    // load from global to shared memory in coalesced manner
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
      floatx16 c = {0};
      c = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
          *reinterpret_cast<long *>(a), *reinterpret_cast<long *>(b), c, 0, 0,
          0);
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

at::Tensor fp8_mm(at::Tensor A, at::Tensor B, at::Tensor A_scale,
                  at::Tensor B_scale, at::Tensor C) {
  size_t N = A.size(0), K = A.size(1), M = B.size(0);

  const size_t BK = 16;
  const size_t BN = 64;
  const size_t BM = 64;
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
    extra_cuda_cflags=["-O3", "--offload-arch=gfx942", "-std=c++20"],
)


def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    return module.fp8_mm(a, b, a_scale, b_scale, c)
  
  #Geometric mean (µs): 362

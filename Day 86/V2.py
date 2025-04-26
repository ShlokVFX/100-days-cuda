#!POPCORN leaderboard amd-fp8-mm
#!POPCORN gpus MI300

import os
import sys
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

// This kernel maintains the original algorithm structure but adds targeted optimizations
template <const size_t BN, const size_t BK, const size_t BM,
          const size_t WITERN, const size_t WITERM>
__global__ void fp8_mm_kernel(const __hip_fp8_e4m3_fnuz *A,
                              const __hip_fp8_e4m3_fnuz *B,
                              const float *A_scale, const float *B_scale,
                              __hip_bfloat16 *C, size_t N, size_t K, size_t M) {
  using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

  static constexpr size_t VECTOR_SIZE = 4;
  static constexpr size_t WARPSIZE = 64;
  static constexpr size_t WN = 32 * WITERN;
  static constexpr size_t WM = 32 * WITERM;
  static constexpr size_t numThreads = (BN * BM) / (16 * WITERN * WITERM);
  static constexpr size_t SUBBN = BN / VECTOR_SIZE;
  static constexpr size_t SUBBM = BM / VECTOR_SIZE;
  static constexpr size_t strideA = numThreads / SUBBN;
  static constexpr size_t strideB = numThreads / SUBBM;

  size_t rowOffsetC = blockIdx.y * BN;
  size_t colOffsetC = blockIdx.x * BM;
  size_t colOffsetA = rowOffsetC;
  size_t colOffsetB = colOffsetC;
  size_t M_scale = cdiv(M, BLOCK_DIM);

  static_assert(numThreads % BN == 0, "BN should be a multiple of numThreads");
  static_assert(numThreads % BM == 0, "BM should be a multiple of numThreads");
  static_assert(BK <= 128 && BM <= 128, "Range above 128 is not supported");

  size_t innerColA = threadIdx.x % SUBBN;
  size_t innerRowA = threadIdx.x / SUBBN;
  size_t innerColB = threadIdx.x % SUBBM;
  size_t innerRowB = threadIdx.x / SUBBM;

  size_t laneIdx = threadIdx.x % WARPSIZE;
  size_t warpIdx = threadIdx.x / WARPSIZE;
  size_t warpColOffset = (warpIdx % (BM / WM)) * WM;
  size_t warpRowOffset = (warpIdx / (BM / WM)) * WN;
  size_t warpX = laneIdx % 32;
  size_t warpY = laneIdx / 32;

  __shared__ __hip_fp8_e4m3_fnuz As[BK][BN], Bs[BK][BM];
  __shared__ float Ws[BN + 1];

  __hip_fp8_e4m3_fnuz a[WITERN][8], b[WITERM][8];
  floatx16 d[WITERN][WITERM] = {0};
  float b_scale;

  for (size_t tileOffset = 0; tileOffset < K; tileOffset += BK) {
    // load from global to shared memory in coalesced manner
    for (size_t innerRowOffsetA = 0; innerRowOffsetA < BK;
         innerRowOffsetA += strideA) {
      if ((tileOffset + innerRowOffsetA + innerRowA) < K &&
          (colOffsetA + innerColA * VECTOR_SIZE) < N &&
          (innerRowOffsetA + innerRowA) < BK) {
        *reinterpret_cast<float *>(
            &As[innerRowOffsetA + innerRowA][innerColA * VECTOR_SIZE]) =
            *reinterpret_cast<const float *>(
                &A[(tileOffset + innerRowOffsetA + innerRowA) * N +
                   (colOffsetA + innerColA * VECTOR_SIZE)]);
      } else if ((innerRowOffsetA + innerRowA) < BK) {
        *reinterpret_cast<float *>(
            &As[innerRowOffsetA + innerRowA][innerColA * VECTOR_SIZE]) = 0.0f;
      }
    }
    if (threadIdx.x < BN) {
      *reinterpret_cast<float4 *>(&Ws[threadIdx.x * VECTOR_SIZE]) =
          *reinterpret_cast<const float4 *>(
              &A_scale[(tileOffset / BLOCK_DIM) * N +
                       (colOffsetA + threadIdx.x * VECTOR_SIZE)]);
    }
    for (size_t innerRowOffsetB = 0; innerRowOffsetB < BK;
         innerRowOffsetB += strideB) {
      if ((tileOffset + innerRowOffsetB + innerRowB) < K &&
          (colOffsetB + innerColB * VECTOR_SIZE) < M &&
          (innerRowOffsetB + innerRowB) < BK) {
        *reinterpret_cast<float *>(
            &Bs[innerRowOffsetB + innerRowB][innerColB * VECTOR_SIZE]) =
            *reinterpret_cast<const float *>(
                &B[(tileOffset + innerRowOffsetB + innerRowB) * M +
                   (colOffsetB + innerColB * VECTOR_SIZE)]);
      } else if ((innerRowOffsetB + innerRowB) < BK &&
                 (innerColB * VECTOR_SIZE) < BM) {
        *reinterpret_cast<float *>(
            &Bs[innerRowOffsetB + innerRowB][innerColB * VECTOR_SIZE]) = 0.0f;
      }
    }
    if (threadIdx.x == numThreads - 1) { // load using different warp
      Ws[BN] = B_scale[(tileOffset / BLOCK_DIM) * M_scale +
                       (colOffsetB / BLOCK_DIM)];
    }

    __syncthreads();

    b_scale = Ws[BN];
    floatx16 c = {0};
    
    for (size_t BKOffset = 0; BKOffset < BK; BKOffset += 16) {
      for (size_t wn = 0; wn < WITERN; ++wn) {
        for (size_t i = 0; i < 8; ++i) {
          a[wn][i] =
              As[BKOffset + warpY * 8 + i][warpRowOffset + wn * 32 + warpX];
        }
      }
      for (size_t wm = 0; wm < WITERM; ++wm) {
        for (size_t i = 0; i < 8; ++i) {
          b[wm][i] =
              Bs[BKOffset + warpY * 8 + i][warpColOffset + wm * 32 + warpX];
        }
      }
      
      for (size_t wn = 0; wn < WITERN; ++wn) {
        for (size_t wm = 0; wm < WITERM; ++wm) {
          c = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
              *reinterpret_cast<long *>(a[wn]),
              *reinterpret_cast<long *>(b[wm]), c, 0, 0, 0);
        }
      }
      
      for (size_t wn = 0; wn < WITERN; ++wn) {
        for (size_t wm = 0; wm < WITERM; ++wm) {
          for (size_t j = 0; j < 4; ++j) {
            for (size_t i = 0; i < 4; ++i) {
              d[wn][wm][i + j * 4] +=
                  c[i + j * 4] *
                  Ws[warpRowOffset + wn * 32 + j * 8 + warpY * 4 + i] * b_scale;
            }
          }
        }
      }
    }

    __syncthreads();
  }

  for (size_t wn = 0; wn < WITERN; ++wn) {
    for (size_t wm = 0; wm < WITERM; ++wm) {
      for (size_t j = 0; j < 4; ++j) {
        for (size_t i = 0; i < 4; ++i) {
          if ((rowOffsetC + warpRowOffset + wn * 32 + j * 8 + warpY * 4 + i) <
                  N &&
              (colOffsetC + warpColOffset + wm * 32 + warpX) < M)
            C[(rowOffsetC + warpRowOffset + wn * 32 + j * 8 + warpY * 4 + i) *
                  M +
              (colOffsetC + warpColOffset + wm * 32 + warpX)] =
                (__hip_bfloat16)d[wn][wm][i + j * 4];
        }
      }
    }
  }
}

// Optimized version for large batch processing
template <const size_t BN, const size_t BK, const size_t BM>
__global__ void fp8_mm_kernel_batched(const __hip_fp8_e4m3_fnuz *A,
                                    const __hip_fp8_e4m3_fnuz *B,
                                    const float *A_scale, const float *B_scale,
                                    __hip_bfloat16 *C, size_t N, size_t K, size_t M) {
  using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

  static constexpr size_t VECTOR_SIZE = 4;
  static constexpr size_t numThreads = (BN * BM) / 16;
  static constexpr size_t SUBBN = BN / VECTOR_SIZE;
  static constexpr size_t SUBBM = BM / VECTOR_SIZE;
  static constexpr size_t strideA = numThreads / SUBBN;
  static constexpr size_t strideB = numThreads / SUBBM;

  size_t rowOffsetC = blockIdx.y * BN;
  size_t colOffsetC = blockIdx.x * BM;
  size_t colOffsetA = rowOffsetC;
  size_t colOffsetB = colOffsetC;
  size_t M_scale = cdiv(M, BLOCK_DIM);

  size_t innerColA = threadIdx.x % SUBBN;
  size_t innerRowA = threadIdx.x / SUBBN;
  size_t innerColB = threadIdx.x % SUBBM;
  size_t innerRowB = threadIdx.x / SUBBM;

  size_t laneIdx = threadIdx.x % 64;
  size_t warpIdx = threadIdx.x / 64;
  size_t warpX = laneIdx % 32;
  size_t warpY = laneIdx / 32;
  size_t warpColOffset = (warpIdx % (BM / 32)) * 32;
  size_t warpRowOffset = (warpIdx / (BM / 32)) * 32;

  __shared__ __hip_fp8_e4m3_fnuz As[BK][BN], Bs[BK][BM];
  __shared__ float Ws[BN + 1];

  __hip_fp8_e4m3_fnuz a[8], b[8];
  floatx16 d = {0};
  float b_scale;

  for (size_t tileOffset = 0; tileOffset < K; tileOffset += BK) {
    // load from global to shared memory in coalesced manner
    for (size_t innerRowOffsetA = 0; innerRowOffsetA < BK;
         innerRowOffsetA += strideA) {
      if ((tileOffset + innerRowOffsetA + innerRowA) < K &&
          (colOffsetA + innerColA * VECTOR_SIZE) < N &&
          (innerRowOffsetA + innerRowA) < BK) {
        *reinterpret_cast<float *>(
            &As[innerRowOffsetA + innerRowA][innerColA * VECTOR_SIZE]) =
            *reinterpret_cast<const float *>(
                &A[(tileOffset + innerRowOffsetA + innerRowA) * N +
                   (colOffsetA + innerColA * VECTOR_SIZE)]);
      } else if ((innerRowOffsetA + innerRowA) < BK) {
        *reinterpret_cast<float *>(
            &As[innerRowOffsetA + innerRowA][innerColA * VECTOR_SIZE]) = 0.0f;
      }
    }
    if (threadIdx.x < BN) {
      *reinterpret_cast<float4 *>(&Ws[threadIdx.x * VECTOR_SIZE]) =
          *reinterpret_cast<const float4 *>(
              &A_scale[(tileOffset / BLOCK_DIM) * N +
                       (colOffsetA + threadIdx.x * VECTOR_SIZE)]);
    }
    for (size_t innerRowOffsetB = 0; innerRowOffsetB < BK;
         innerRowOffsetB += strideB) {
      if ((tileOffset + innerRowOffsetB + innerRowB) < K &&
          (colOffsetB + innerColB * VECTOR_SIZE) < M &&
          (innerRowOffsetB + innerRowB) < BK) {
        *reinterpret_cast<float *>(
            &Bs[innerRowOffsetB + innerRowB][innerColB * VECTOR_SIZE]) =
            *reinterpret_cast<const float *>(
                &B[(tileOffset + innerRowOffsetB + innerRowB) * M +
                   (colOffsetB + innerColB * VECTOR_SIZE)]);
      } else if ((innerRowOffsetB + innerRowB) < BK &&
                 (innerColB * VECTOR_SIZE) < BM) {
        *reinterpret_cast<float *>(
            &Bs[innerRowOffsetB + innerRowB][innerColB * VECTOR_SIZE]) = 0.0f;
      }
    }
    if (threadIdx.x == numThreads - 1) { // load using different warp
      Ws[BN] = B_scale[(tileOffset / BLOCK_DIM) * M_scale +
                       (colOffsetB / BLOCK_DIM)];
    }

    __syncthreads();

    b_scale = Ws[BN];
    
    for (size_t BKOffset = 0; BKOffset < BK; BKOffset += 16) {
      for (size_t i = 0; i < 8; ++i) {
        a[i] = As[BKOffset + warpY * 8 + i][warpRowOffset + warpX];
        b[i] = Bs[BKOffset + warpY * 8 + i][warpColOffset + warpX];
      }
      
      floatx16 c = {0};
      c = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
          *reinterpret_cast<long *>(a),
          *reinterpret_cast<long *>(b), c, 0, 0, 0);
          
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

  const size_t BK = 128;
  const size_t BN = 128;
  const size_t BM = 128;

  // Choose kernel based on matrix dimensions
  if (N >= 1024 && M >= 1024) {
    // Batched processing for large matrices
    dim3 numThreads((BN * BM) / 16);
    dim3 numBlocks(cdiv(M, BM), cdiv(N, BN));
    fp8_mm_kernel_batched<BN, BK, BM><<<numBlocks, numThreads>>>(
        (__hip_fp8_e4m3_fnuz *)A.data_ptr(), (__hip_fp8_e4m3_fnuz *)B.data_ptr(),
        A_scale.data_ptr<float>(), B_scale.data_ptr<float>(),
        (__hip_bfloat16 *)C.data_ptr(), N, K, M);
  } else {
    // Original kernel with modest optimization for smaller matrices
    const size_t WITERN = 1;
    const size_t WITERM = 1;
    dim3 numThreads((BN * BM) / (16 * WITERN * WITERM));
    dim3 numBlocks(cdiv(M, BM), cdiv(N, BN));
    fp8_mm_kernel<BN, BK, BM, WITERN, WITERM><<<numBlocks, numThreads>>>(
        (__hip_fp8_e4m3_fnuz *)A.data_ptr(), (__hip_fp8_e4m3_fnuz *)B.data_ptr(),
        A_scale.data_ptr<float>(), B_scale.data_ptr<float>(),
        (__hip_bfloat16 *)C.data_ptr(), N, K, M);
  }
  
  return C;
}
"""

cpp_src = r"""
at::Tensor fp8_mm(at::Tensor A, at::Tensor B, at::Tensor A_scale,
                  at::Tensor B_scale, at::Tensor C);
"""

if sys.stdout is None:
    sys.stdout = open("/dev/stdout", "w")
if sys.stderr is None:
    sys.stderr = open("/dev/stderr", "w")

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
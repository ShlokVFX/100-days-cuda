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

__host__ __device__ __forceinline__ int cdiv(int a, int b) {
  return (a + b - 1) / b;
}

#define BLOCK_DIM 128

template <const uint32_t BN, const uint32_t BK, const uint32_t BM,
          const uint32_t WITERN, const uint32_t WITERM, const uint32_t SM_COUNT>
          __global__ __launch_bounds__(1024, 4)
__global__ void fp8_mm_kernel_N(const __hip_fp8_e4m3_fnuz *A, const __hip_fp8_e4m3_fnuz *B,
                const float *A_scale, const float *B_scale, __hip_bfloat16 *C,
                uint32_t N, uint32_t K, uint32_t M) {
  using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

  static constexpr uint32_t VECTOR_SIZE = 4;
  static constexpr uint32_t WARPSIZE = 64;
  static constexpr uint32_t WN = 32 * WITERN;
  static constexpr uint32_t WM = 32 * WITERM;
  static constexpr uint32_t numThreads = (BN * BM) / (16 * WITERN * WITERM);
  static constexpr uint32_t SUBBN = BN / VECTOR_SIZE;
  static constexpr uint32_t SUBBM = BM / VECTOR_SIZE;
  static constexpr uint32_t strideA = numThreads / SUBBN;
  static constexpr uint32_t strideB = numThreads / SUBBM;

  static_assert(numThreads % BN == 0, "BN should be a multiple of numThreads");
  static_assert(numThreads % BM == 0, "BM should be a multiple of numThreads");
  static_assert(BK <= 128 && BM <= 128, "Range above 128 is not supported");

  uint32_t numTiles = cdiv(N, BN) * cdiv(M, BM);

  for (uint32_t tileIdx = blockIdx.x; tileIdx < numTiles; tileIdx += SM_COUNT) {
    uint32_t rowOffsetC = (tileIdx % cdiv(N, BN)) * BN;
    uint32_t colOffsetC = (tileIdx / cdiv(N, BN)) * BM;
    uint32_t colOffsetA = rowOffsetC;
    uint32_t colOffsetB = colOffsetC;
    uint32_t M_scale = cdiv(M, BLOCK_DIM);

    uint32_t innerColA = threadIdx.x % SUBBN;
    uint32_t innerRowA = threadIdx.x / SUBBN;
    uint32_t innerColB = threadIdx.x % SUBBM;
    uint32_t innerRowB = threadIdx.x / SUBBM;

    uint32_t laneIdx = threadIdx.x % WARPSIZE;
    uint32_t warpIdx = threadIdx.x / WARPSIZE;
    uint32_t warpColOffset = (warpIdx % (BM / WM)) * WM;
    uint32_t warpRowOffset = (warpIdx / (BM / WM)) * WN;
    uint32_t warpX = laneIdx % 32;
    uint32_t warpY = laneIdx / 32;

    __shared__ __hip_fp8_e4m3_fnuz As[BK][BN], Bs[BK][BM+4];
    __shared__ float Ws[BN + 1];

    __hip_fp8_e4m3_fnuz a[WITERN][8], b[WITERM][8];
    floatx16 d[WITERN][WITERM] = {0};

    for (uint32_t tileOffset = 0; tileOffset < K; tileOffset += BK) {
      // load from global to shared memory in coalesced manner
      for (uint32_t innerRowOffsetA = 0; innerRowOffsetA < BK;
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
      if (threadIdx.x < SUBBN) {
        *reinterpret_cast<float4 *>(&Ws[threadIdx.x * VECTOR_SIZE]) =
            *reinterpret_cast<const float4 *>(
                &A_scale[(tileOffset / BLOCK_DIM) * N +
                         (colOffsetA + threadIdx.x * VECTOR_SIZE)]);
      }
      for (uint32_t innerRowOffsetB = 0; innerRowOffsetB < BK;
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

      float b_scale = Ws[BN];
      floatx16 c[WITERN][WITERM] = {0};
      for (uint32_t BKOffset = 0; BKOffset < BK; BKOffset += 16) {
        for (uint32_t wn = 0; wn < WITERN; ++wn) {
          for (uint32_t i = 0; i < 8; ++i) {
            a[wn][i] =
                As[BKOffset + warpY * 8 + i][warpRowOffset + wn * 32 + warpX];
          }
        }
        for (uint32_t wm = 0; wm < WITERM; ++wm) {
          for (uint32_t i = 0; i < 8; ++i) {
            b[wm][i] =
                Bs[BKOffset + warpY * 8 + i][warpColOffset + wm * 32 + warpX];
          }
        }
        for (uint32_t wn = 0; wn < WITERN; ++wn) {
          for (uint32_t wm = 0; wm < WITERM; ++wm) {
            c[wn][wm] = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
                *reinterpret_cast<long *>(a[wn]),
                *reinterpret_cast<long *>(b[wm]), c[wn][wm], 0, 0, 0);
          }
        }
      }
      for (uint32_t wn = 0; wn < WITERN; ++wn) {
        for (uint32_t wm = 0; wm < WITERM; ++wm) {
          for (uint32_t j = 0; j < 4; ++j) {
            for (uint32_t i = 0; i < 4; ++i) {
              d[wn][wm][i + j * 4] +=
                  c[wn][wm][i + j * 4] *
                  Ws[warpRowOffset + wn * 32 + j * 8 + warpY * 4 + i] * b_scale;
            }
          }
        }
      }

      __syncthreads();
    }

//Write back
 for (uint32_t wn = 0; wn < WITERN; ++wn) {
    for (uint32_t wm = 0; wm < WITERM; ++wm) {
      for (uint32_t j = 0; j < 4; ++j) {
        for (uint32_t i = 0; i < 4; ++i) {
          uint32_t outRow = rowOffsetC + warpRowOffset + wn * 32 + j * 8 + warpY * 4 + i;
          uint32_t outCol = colOffsetC + warpColOffset + wm * 32 + warpX;
          if (outRow < N && outCol < M) {
            uint32_t idx = outRow * M + outCol;
  
            // 1) extract the float accumulator
            float accum = d[wn][wm][i + j * 4];
            
            // 2) convert to bfloat16
            __hip_bfloat16 bf16 = static_cast<__hip_bfloat16>(accum);

            // 3) pack the raw 16-bit representation
            union { __hip_bfloat16 bf; __hip_fp8_e4m3_fnuz u; } pack;
            pack.bf = bf16;
            __hip_fp8_e4m3_fnuz val16 = pack.u;
  
            // 4) pointer to C as 16-bit integer array
            __hip_fp8_e4m3_fnuz* outPtr = reinterpret_cast<__hip_fp8_e4m3_fnuz*>(C) + idx;
  
            // 5) non-temporal (cache-bypass) store
            __builtin_nontemporal_store(val16, outPtr);
          }
        }
      }
    }
  }
}
}

template <const uint32_t BN, const uint32_t BK, const uint32_t BM,
          const uint32_t WITERN, const uint32_t WITERM, const uint32_t SM_COUNT>
          __global__ __launch_bounds__(1024, 4)
__global__ void fp8_mm_kernel_M(const __hip_fp8_e4m3_fnuz *A, const __hip_fp8_e4m3_fnuz *B,
                const float *A_scale, const float *B_scale, __hip_bfloat16 *C,
                uint32_t N, uint32_t K, uint32_t M) {
  using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

  static constexpr uint32_t VECTOR_SIZE = 4;
  static constexpr uint32_t WARPSIZE = 64;
  static constexpr uint32_t WN = 32 * WITERN;
  static constexpr uint32_t WM = 32 * WITERM;
  static constexpr uint32_t numThreads = (BN * BM) / (16 * WITERN * WITERM);
  static constexpr uint32_t SUBBN = BN / VECTOR_SIZE;
  static constexpr uint32_t SUBBM = BM / VECTOR_SIZE;
  static constexpr uint32_t strideA = numThreads / SUBBN;
  static constexpr uint32_t strideB = numThreads / SUBBM;

  static_assert(numThreads % BN == 0, "BN should be a multiple of numThreads");
  static_assert(numThreads % BM == 0, "BM should be a multiple of numThreads");
  static_assert(BK <= 128 && BM <= 128, "Range above 128 is not supported");

  uint32_t numTiles = cdiv(N, BN) * cdiv(M, BM);

  for (uint32_t tileIdx = blockIdx.x; tileIdx < numTiles; tileIdx += SM_COUNT) {
    uint32_t rowOffsetC = (tileIdx / cdiv(M, BM)) * BN;
    uint32_t colOffsetC = (tileIdx % cdiv(M, BM)) * BM;
    uint32_t colOffsetA = rowOffsetC;
    uint32_t colOffsetB = colOffsetC;
    uint32_t M_scale = cdiv(M, BLOCK_DIM);

    uint32_t innerColA = threadIdx.x % SUBBN;
    uint32_t innerRowA = threadIdx.x / SUBBN;
    uint32_t innerColB = threadIdx.x % SUBBM;
    uint32_t innerRowB = threadIdx.x / SUBBM;

    uint32_t laneIdx = threadIdx.x % WARPSIZE;
    uint32_t warpIdx = threadIdx.x / WARPSIZE;
    uint32_t warpColOffset = (warpIdx % (BM / WM)) * WM;
    uint32_t warpRowOffset = (warpIdx / (BM / WM)) * WN;
    uint32_t warpX = laneIdx % 32;
    uint32_t warpY = laneIdx / 32;

    __shared__ __hip_fp8_e4m3_fnuz As[BK][BN], Bs[BK][BM+4];
    __shared__ float Ws[BN + 1];

    __hip_fp8_e4m3_fnuz a[WITERN][8], b[WITERM][8];
    floatx16 d[WITERN][WITERM] = {0};

    for (uint32_t tileOffset = 0; tileOffset < K; tileOffset += BK) {
      // load from global to shared memory in coalesced manner
      for (uint32_t innerRowOffsetA = 0; innerRowOffsetA < BK;
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
      if (threadIdx.x < SUBBN) {
        *reinterpret_cast<float4 *>(&Ws[threadIdx.x * VECTOR_SIZE]) =
            *reinterpret_cast<const float4 *>(
                &A_scale[(tileOffset / BLOCK_DIM) * N +
                         (colOffsetA + threadIdx.x * VECTOR_SIZE)]);
      }
      for (uint32_t innerRowOffsetB = 0; innerRowOffsetB < BK;
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

      float b_scale = Ws[BN];
      floatx16 c[WITERN][WITERM] = {0};
      for (uint32_t BKOffset = 0; BKOffset < BK; BKOffset += 16) {
        for (uint32_t wn = 0; wn < WITERN; ++wn) {
          for (uint32_t i = 0; i < 8; ++i) {
            a[wn][i] =
                As[BKOffset + warpY * 8 + i][warpRowOffset + wn * 32 + warpX];
          }
        }
        for (uint32_t wm = 0; wm < WITERM; ++wm) {
          for (uint32_t i = 0; i < 8; ++i) {
            b[wm][i] =
                Bs[BKOffset + warpY * 8 + i][warpColOffset + wm * 32 + warpX];
          }
        }
        for (uint32_t wn = 0; wn < WITERN; ++wn) {
          for (uint32_t wm = 0; wm < WITERM; ++wm) {
            c[wn][wm] = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
                *reinterpret_cast<long *>(a[wn]),
                *reinterpret_cast<long *>(b[wm]), c[wn][wm], 0, 0, 0);
          }
        }
      }
      for (uint32_t wn = 0; wn < WITERN; ++wn) {
        for (uint32_t wm = 0; wm < WITERM; ++wm) {
          for (uint32_t j = 0; j < 4; ++j) {
            for (uint32_t i = 0; i < 4; ++i) {
              d[wn][wm][i + j * 4] +=
                  c[wn][wm][i + j * 4] *
                  Ws[warpRowOffset + wn * 32 + j * 8 + warpY * 4 + i] * b_scale;
            }
          }
        }
      }

      __syncthreads();
    }

//Write back
 for (uint32_t wn = 0; wn < WITERN; ++wn) {
    for (uint32_t wm = 0; wm < WITERM; ++wm) {
      for (uint32_t j = 0; j < 4; ++j) {
        for (uint32_t i = 0; i < 4; ++i) {
          uint32_t outRow = rowOffsetC + warpRowOffset + wn * 32 + j * 8 + warpY * 4 + i;
          uint32_t outCol = colOffsetC + warpColOffset + wm * 32 + warpX;
          if (outRow < N && outCol < M) {
            uint32_t idx = outRow * M + outCol;
  
            // 1) extract the float accumulator
            float accum = d[wn][wm][i + j * 4];
            
            // 2) convert to bfloat16
            __hip_bfloat16 bf16 = static_cast<__hip_bfloat16>(accum);

            // 3) pack the raw 16-bit representation
            union { __hip_bfloat16 bf; __hip_fp8_e4m3_fnuz u; } pack;
            pack.bf = bf16;
            __hip_fp8_e4m3_fnuz val16 = pack.u;
  
            // 4) pointer to C as 16-bit integer array
            __hip_fp8_e4m3_fnuz* outPtr = reinterpret_cast<__hip_fp8_e4m3_fnuz*>(C) + idx;
  
            // 5) non-temporal (cache-bypass) store
            __builtin_nontemporal_store(val16, outPtr);
          }
        }
      }
    }
  }
}

}


at::Tensor fp8_mm(at::Tensor A, at::Tensor B, at::Tensor A_scale,
                  at::Tensor B_scale, at::Tensor C) {
  int N = A.size(0), K = A.size(1), M = B.size(0);

  const int BK = 128;
  const int BN = 128;
  const int BM = 128;
  const int WITERN = 1;
  const int WITERM = 1;
  const int SM_COUNT = 304;
  dim3 numThreads((BN * BM) / (16 * WITERN * WITERM));
  dim3 numBlocks(SM_COUNT);
  if (N > M) {
      fp8_mm_kernel_N<BN, BK, BM, WITERN, WITERM, SM_COUNT><<<numBlocks, numThreads>>>(
          (__hip_fp8_e4m3_fnuz *)A.data_ptr(), (__hip_fp8_e4m3_fnuz *)B.data_ptr(),
          A_scale.data_ptr<float>(), B_scale.data_ptr<float>(),
          (__hip_bfloat16 *)C.data_ptr(), N, K, M);
  } else {
      fp8_mm_kernel_M<BN, BK, BM, WITERN, WITERM, SM_COUNT><<<numBlocks, numThreads>>>(
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
    extra_cuda_cflags=[
        "-Ofast",
        "--offload-arch=gfx942",
        "-std=c++20",
        "-ffp-contract=fast",
        "-lhip_hcc",
         "-mcumode",
    ],
)

def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    return module.fp8_mm(a, b, a_scale, b_scale, c)

#!POPCORN leaderboard amd-fp32-mm
#!POPCORN gpus MI300

import os
import sys
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "clang++"

cuda_src = r"""
#include <iostream>
#include <hip/hip_runtime.h>

__host__ __device__ __forceinline__ int cdiv(int a, int b) {
  return (a + b - 1) / b;
}

#define BLOCK_DIM 128

// Generic enum for tile index calculation strategy
enum class TileIndexingStrategy {
  M_MAJOR, // tileIdx / cdiv(M, BM) gives row, tileIdx % cdiv(M, BM) gives col
  N_MAJOR  // tileIdx % cdiv(N, BN) gives row, tileIdx / cdiv(N, BN) gives col
};

template <const uint32_t BN, const uint32_t BK, const uint32_t BM,
          const uint32_t WITERN, const uint32_t WITERM, const uint32_t SM_COUNT,
          const TileIndexingStrategy strategy>
__global__
__attribute__((amdgpu_flat_work_group_size(0,0)))
__launch_bounds__(1024, 4)
void fp32_mm_kernel(const float *A, const float *B, float *C,
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
  uint32_t rowOffsetC, colOffsetC;

  for (uint32_t tileIdx = blockIdx.x; tileIdx < numTiles; tileIdx += SM_COUNT) {
    // Compute tile indices differently based on strategy
    if constexpr (strategy == TileIndexingStrategy::M_MAJOR) {
      rowOffsetC = (tileIdx / cdiv(M, BM)) * BN;
      colOffsetC = (tileIdx % cdiv(M, BM)) * BM;
    } else { // N_MAJOR
      rowOffsetC = (tileIdx % cdiv(N, BN)) * BN;
      colOffsetC = (tileIdx / cdiv(N, BN)) * BM;
    }
    
    uint32_t colOffsetA = rowOffsetC;
    uint32_t colOffsetB = colOffsetC;

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

    // Double-buffering setup: two sets of shared memory buffers
    __shared__ float As1[BK][BN+8], Bs1[BK][BM+8];
    __shared__ float As2[BK][BN+8], Bs2[BK][BM+8];
    auto As = As1, Bs = Bs1;
    auto Ast = As2, Bst = Bs2;

    // For FP32, we directly load float arrays
    float a[WITERN][8], b[WITERM][8];
    floatx16 d[WITERN][WITERM] = {0};

    // Initial load: global memory -> shared memory
    for (uint32_t innerRowOffsetA = 0; innerRowOffsetA < BK;
         innerRowOffsetA += strideA) {
      if ((innerRowOffsetA + innerRowA) < K &&
          (colOffsetA + innerColA * VECTOR_SIZE) < N &&
          (innerRowOffsetA + innerRowA) < BK) {
        *reinterpret_cast<float4 *>(
            &As[innerRowOffsetA + innerRowA][innerColA * VECTOR_SIZE]) =
            *reinterpret_cast<const float4 *>(
                &A[(innerRowOffsetA + innerRowA) * N +
                   (colOffsetA + innerColA * VECTOR_SIZE)]);
      } else if ((innerRowOffsetA + innerRowA) < BK) {
        *reinterpret_cast<float4 *>(
            &As[innerRowOffsetA + innerRowA][innerColA * VECTOR_SIZE]) = 
            make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
    }

    for (uint32_t innerRowOffsetB = 0; innerRowOffsetB < BK;
         innerRowOffsetB += strideB) {
      if ((innerRowOffsetB + innerRowB) < K &&
          (colOffsetB + innerColB * VECTOR_SIZE) < M &&
          (innerRowOffsetB + innerRowB) < BK) {
        *reinterpret_cast<float4 *>(
            &Bs[innerRowOffsetB + innerRowB][innerColB * VECTOR_SIZE]) =
            *reinterpret_cast<const float4 *>(
                &B[(innerRowOffsetB + innerRowB) * M +
                   (colOffsetB + innerColB * VECTOR_SIZE)]);
      } else if ((innerRowOffsetB + innerRowB) < BK &&
                 (innerColB * VECTOR_SIZE) < BM) {
        *reinterpret_cast<float4 *>(
            &Bs[innerRowOffsetB + innerRowB][innerColB * VECTOR_SIZE]) = 
            make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
    }

    __syncthreads();

    // Temporary storage for next tile
    float4 Att[2], Btt[2];
    
    // Main computation loop with double buffering
    for (uint32_t tileOffset = BK; tileOffset < K + BK; tileOffset += BK) {
      // Load next block (if within bounds)
      if (tileOffset < K) {
        // Load next A tile
        for (uint32_t innerRowOffsetA = 0; innerRowOffsetA < BK;
             innerRowOffsetA += strideA) {
          if ((tileOffset + innerRowOffsetA + innerRowA) < K &&
              (colOffsetA + innerColA * VECTOR_SIZE) < N &&
              (innerRowOffsetA + innerRowA) < BK) {
            Att[innerRowOffsetA / strideA] = *reinterpret_cast<const float4 *>(
                &A[(tileOffset + innerRowOffsetA + innerRowA) * N +
                   (colOffsetA + innerColA * VECTOR_SIZE)]);
          } else if ((innerRowOffsetA + innerRowA) < BK) {
            Att[innerRowOffsetA / strideA] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
          }
        }
        
        // Load next B tile
        for (uint32_t innerRowOffsetB = 0; innerRowOffsetB < BK;
             innerRowOffsetB += strideB) {
          if ((tileOffset + innerRowOffsetB + innerRowB) < K &&
              (colOffsetB + innerColB * VECTOR_SIZE) < M &&
              (innerRowOffsetB + innerRowB) < BK) {
            Btt[innerRowOffsetB / strideB] = *reinterpret_cast<const float4 *>(
                &B[(tileOffset + innerRowOffsetB + innerRowB) * M +
                   (colOffsetB + innerColB * VECTOR_SIZE)]);
          } else if ((innerRowOffsetB + innerRowB) < BK &&
                     (innerColB * VECTOR_SIZE) < BM) {
            Btt[innerRowOffsetB / strideB] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
          }
        }
      }

      // Compute current block
      floatx16 c[WITERN][WITERM] = {0};
      
      // Process BK in chunks of 16 (matrix multiplication using MFMA)
      for (uint32_t BKOffset = 0; BKOffset < BK; BKOffset += 16) {
        // Load A matrix elements
        for (uint32_t wn = 0; wn < WITERN; ++wn) {
          for (uint32_t i = 0; i < 8; ++i) {
            a[wn][i] =
                As[BKOffset + warpY * 8 + i][warpRowOffset + wn * 32 + warpX];
          }
        }
        
        // Load B matrix elements
        for (uint32_t wm = 0; wm < WITERM; ++wm) {
          for (uint32_t i = 0; i < 8; ++i) {
            b[wm][i] =
                Bs[BKOffset + warpY * 8 + i][warpColOffset + wm * 32 + warpX];
          }
        }
        
        // Matrix multiply using AMD MFMA instruction for FP32
        for (uint32_t wn = 0; wn < WITERN; ++wn) {
          for (uint32_t wm = 0; wm < WITERM; ++wm) {
            c[wn][wm] = __builtin_amdgcn_mfma_f32_32x32x8_f32(
                *reinterpret_cast<float4 *>(a[wn]),
                *reinterpret_cast<float4 *>(b[wm]), c[wn][wm], 0, 0, 0);
          }
        }
      }
      
      // Accumulate results
      for (uint32_t wn = 0; wn < WITERN; ++wn) {
        for (uint32_t wm = 0; wm < WITERM; ++wm) {
          for (uint32_t j = 0; j < 16; ++j) {
            d[wn][wm][j] += c[wn][wm][j];
          }
        }
      }

      // Store loaded data to shared memory (for next iteration)
      if (tileOffset < K) {
        // Store A
        for (uint32_t innerRowOffsetA = 0; innerRowOffsetA < BK;
             innerRowOffsetA += strideA) {
          if ((innerRowOffsetA + innerRowA) < BK) {
            *reinterpret_cast<float4 *>(
                &Ast[innerRowOffsetA + innerRowA][innerColA * VECTOR_SIZE]) =
                Att[innerRowOffsetA / strideA];
          }
        }
        
        // Store B
        for (uint32_t innerRowOffsetB = 0; innerRowOffsetB < BK;
             innerRowOffsetB += strideB) {
          if ((innerRowOffsetB + innerRowB) < BK &&
              (innerColB * VECTOR_SIZE) < BM) {
            *reinterpret_cast<float4 *>(
                &Bst[innerRowOffsetB + innerRowB][innerColB * VECTOR_SIZE]) =
                Btt[innerRowOffsetB / strideB];
          }
        }
      }

      __syncthreads();

      // Swap buffer pointers
      auto tmp = As;
      As = Ast;
      Ast = tmp;
      tmp = Bs;
      Bs = Bst;
      Bst = tmp;
    }

    // Write final results to global memory
    for (uint32_t wn = 0; wn < WITERN; ++wn) {
      for (uint32_t wm = 0; wm < WITERM; ++wm) {
        for (uint32_t j = 0; j < 4; ++j) {
          for (uint32_t i = 0; i < 4; ++i) {
            uint32_t globalRow = rowOffsetC + warpRowOffset + wn * 32 + j * 8 + warpY * 4 + i;
            uint32_t globalCol = colOffsetC + warpColOffset + wm * 32 + warpX;
            if (globalRow < N && globalCol < M) {
              C[globalRow * M + globalCol] = d[wn][wm][i + j * 4];
            }
          }
        }
      }
    }
  }
}

at::Tensor fp32_mm(at::Tensor A, at::Tensor B, at::Tensor C) {
  int N = A.size(0), K = A.size(1), M = B.size(0);

  // Optimized parameter settings for MI300
  const int BK = 64;       // Tile size for K dimension
  const int BN = 128;      // Tile size for N dimension
  const int BM = 128;      // Tile size for M dimension
  const int WITERN = 1;    // Work-items per N
  const int WITERM = 1;    // Work-items per M
  const int SM_COUNT = 304; // Number of compute units on MI300

  dim3 numThreads((BN * BM) / (16 * WITERN * WITERM));
  dim3 numBlocks(SM_COUNT);
  
  // Choose kernel based on matrix dimensions
  if (N > M) {
      fp32_mm_kernel<BN, BK, BM, WITERN, WITERM, SM_COUNT, TileIndexingStrategy::N_MAJOR>
      <<<numBlocks, numThreads>>>(
          A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
          N, K, M);
  } else {
      fp32_mm_kernel<BN, BK, BM, WITERN, WITERM, SM_COUNT, TileIndexingStrategy::M_MAJOR>
      <<<numBlocks, numThreads>>>(
          A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
          N, K, M);
  }
  
  // Check for errors
  hipError_t error = hipGetLastError();
  if (error != hipSuccess) {
    printf("CUDA error: %s\n", hipGetErrorString(error));
  }
  
  return C;
}
"""

cpp_src = r"""
at::Tensor fp32_mm(at::Tensor A, at::Tensor B, at::Tensor C);
"""

if sys.stdout is None:
    sys.stdout = open("/dev/stdout", "w")
if sys.stderr is None:
    sys.stderr = open("/dev/stderr", "w")

module = load_inline(
    name="fp32_mm",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["fp32_mm"],
    verbose=True,
    extra_cuda_cflags=[
        "-Ofast",
        "--offload-arch=gfx942",
        "-std=c++20",
        "-ffp-contract=fast",
    ],
)

def custom_kernel(data: input_t) -> output_t:
    a, b, c = data
    return module.fp32_mm(a, b, c)

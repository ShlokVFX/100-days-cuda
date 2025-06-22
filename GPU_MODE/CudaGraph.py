#!POPCORN leaderboard amd-fp8-mm
#!POPCORN gpus MI300

import os
import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "clang++"

# Pre-compile optimization flags
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.4"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Keep your existing cuda_src exactly as it was in the original file
cuda_src = r"""
#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/amd_detail/amd_hip_fp8.h>

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
          __launch_bounds__(1024,4)
__global__ void
fp8_mm_kernel(const __hip_fp8_e4m3_fnuz *A, const __hip_fp8_e4m3_fnuz *B,
              const float *A_scale, const float *B_scale, __hip_bfloat16 *C,
              uint32_t N, uint32_t K, uint32_t M) {
  using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

  static constexpr uint32_t VECTOR_SIZE = 4;
  static constexpr uint32_t WARPSIZE = 64;
  static constexpr uint32_t WN = 32 * WITERN;
  static constexpr uint32_t WM = 32 * WITERM;
  static constexpr uint32_t numThreads = (BN * BM) / (16 * WITERN * WITERM);
  static constexpr uint32_t strideA = (numThreads / BN) * VECTOR_SIZE;
  static constexpr uint32_t strideB = (numThreads / BM) * VECTOR_SIZE;

  static_assert(numThreads % BN == 0, "BN should be a multiple of numThreads");
  static_assert(numThreads % BM == 0, "BM should be a multiple of numThreads");
  static_assert(BK <= 128 && BM <= 128, "Range above 128 is not supported");

  uint32_t numTiles = cdiv(N, BN) * cdiv(M, BM);
  uint32_t rowOffsetC, colOffsetC;

  for (uint16_t tileIdx = blockIdx.x; tileIdx < numTiles; tileIdx += SM_COUNT) {
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
    uint32_t M_scale = cdiv(M, BLOCK_DIM);

    uint32_t innerColA = threadIdx.x % BN;
    uint32_t innerRowA = (threadIdx.x / BN) * VECTOR_SIZE;
    uint32_t innerColB = threadIdx.x % BM;
    uint32_t innerRowB = (threadIdx.x / BM) * VECTOR_SIZE;

    uint32_t laneIdx = threadIdx.x % WARPSIZE;
    uint32_t warpIdx = threadIdx.x / WARPSIZE;
    uint32_t warpColOffset = (warpIdx % (BM / WM)) * WM;
    uint32_t warpRowOffset = (warpIdx / (BM / WM)) * WN;
    uint32_t warpX = laneIdx % 32;
    uint32_t warpY = laneIdx / 32;

    // Double-buffering setup with array indices
    __shared__ __hip_fp8_e4m3_fnuz As[2][BN][BK + 4], Bs[2][BM][BK + 4];
    __shared__ float Ws[2][BN + 1];

    int curr = 0, next = 1;
    floatx16 d[WITERN][WITERM] = {0};

    // Initial load: global memory -> shared memory
    for (uint16_t innerRowOffsetA = 0; innerRowOffsetA < BK;
         innerRowOffsetA += strideA) {
      if ((innerRowOffsetA + innerRowA) < K && (colOffsetA + innerColA) < N &&
          (innerRowOffsetA + innerRowA) < BK) {
        // Optimized byte packing with improved method
        // Load 4 consecutive bytes and pack them efficiently
        uint8_t x = *reinterpret_cast<const uint8_t *>(
            &A[(innerRowOffsetA + innerRowA) * N + (colOffsetA + innerColA)]);
        uint8_t y = *reinterpret_cast<const uint8_t *>(
            &A[(innerRowOffsetA + innerRowA + 1) * N + (colOffsetA + innerColA)]);
        uint8_t z = *reinterpret_cast<const uint8_t *>(
            &A[(innerRowOffsetA + innerRowA + 2) * N + (colOffsetA + innerColA)]);
        uint8_t w = *reinterpret_cast<const uint8_t *>(
            &A[(innerRowOffsetA + innerRowA + 3) * N + (colOffsetA + innerColA)]);

        // Optimized packing with fewer operations and better register usage
        unsigned int pack = ((unsigned int)w << 24) | 
                           ((unsigned int)z << 16) | 
                           ((unsigned int)y << 8) | 
                            (unsigned int)x;

        *reinterpret_cast<unsigned int *>(
            &As[curr][innerColA][innerRowOffsetA + innerRowA]) = pack;
      } else if ((innerRowOffsetA + innerRowA) < BK) {
        *reinterpret_cast<float *>(
            &As[curr][innerColA][innerRowOffsetA + innerRowA]) = 0.0f;
      }
    }
    
    if (threadIdx.x < (BN / VECTOR_SIZE)) {
      *reinterpret_cast<float4 *>(&Ws[curr][threadIdx.x * VECTOR_SIZE]) =
          *reinterpret_cast<const float4 *>(
              &A_scale[(colOffsetA + threadIdx.x * VECTOR_SIZE)]);
    }
    
    for (uint16_t innerRowOffsetB = 0; innerRowOffsetB < BK;
         innerRowOffsetB += strideB) {
      if ((innerRowOffsetB + innerRowB) < K && (colOffsetB + innerColB) < M &&
          (innerRowOffsetB + innerRowB) < BK) {
        // Optimized byte packing for B matrix
        uint8_t x = *reinterpret_cast<const uint8_t *>(
            &B[(innerRowOffsetB + innerRowB) * M + (colOffsetB + innerColB)]);
        uint8_t y = *reinterpret_cast<const uint8_t *>(
            &B[(innerRowOffsetB + innerRowB + 1) * M + (colOffsetB + innerColB)]);
        uint8_t z = *reinterpret_cast<const uint8_t *>(
            &B[(innerRowOffsetB + innerRowB + 2) * M + (colOffsetB + innerColB)]);
        uint8_t w = *reinterpret_cast<const uint8_t *>(
            &B[(innerRowOffsetB + innerRowB + 3) * M + (colOffsetB + innerColB)]);

        // Optimized packing with fewer operations
        unsigned int pack = ((unsigned int)w << 24) | 
                           ((unsigned int)z << 16) | 
                           ((unsigned int)y << 8) | 
                            (unsigned int)x;

        *reinterpret_cast<unsigned int *>(
            &Bs[curr][innerColB][innerRowOffsetB + innerRowB]) = pack;
      } else if ((innerRowOffsetB + innerRowB) < BK) {
        *reinterpret_cast<float *>(
            &Bs[curr][innerColB][innerRowOffsetB + innerRowB]) = 0.0f;
      }
    }
    
    if (threadIdx.x == numThreads - 1) {
      Ws[curr][BN] = B_scale[(colOffsetB / BLOCK_DIM)];
    }

    __syncthreads();

    // Optimize storage for temporary data - better register usage
    uint32_t A_bytes[4][2];  // [byte_index][chunk_index]
    uint32_t B_bytes[4][2];
    float4 A_scale_temp;
    float B_scale_temp;
    
    // Main computation loop with double buffering
    for (uint16_t tileOffset = BK; tileOffset < K + BK; tileOffset += BK) {
      // Load next block (if within bounds)
      if (tileOffset < K) {
        // Load next A tile 
        for (uint16_t innerRowOffsetA = 0; innerRowOffsetA < BK;
             innerRowOffsetA += strideA) {
          if ((tileOffset + innerRowOffsetA + innerRowA) < K &&
              (colOffsetA + innerColA) < N &&
              (innerRowOffsetA + innerRowA) < BK) {
            // Store bytes for efficient packing later
            A_bytes[0][innerRowOffsetA / strideA] =
                *reinterpret_cast<const uint8_t *>(
                    &A[(tileOffset + innerRowOffsetA + innerRowA) * N +
                       (colOffsetA + innerColA)]);
            A_bytes[1][innerRowOffsetA / strideA] =
                *reinterpret_cast<const uint8_t *>(
                    &A[(tileOffset + innerRowOffsetA + innerRowA + 1) * N +
                       (colOffsetA + innerColA)]);
            A_bytes[2][innerRowOffsetA / strideA] =
                *reinterpret_cast<const uint8_t *>(
                    &A[(tileOffset + innerRowOffsetA + innerRowA + 2) * N +
                       (colOffsetA + innerColA)]);
            A_bytes[3][innerRowOffsetA / strideA] =
                *reinterpret_cast<const uint8_t *>(
                    &A[(tileOffset + innerRowOffsetA + innerRowA + 3) * N +
                       (colOffsetA + innerColA)]);
          } else if ((innerRowOffsetA + innerRowA) < BK) {
            A_bytes[0][innerRowOffsetA / strideA] = 0;
            A_bytes[1][innerRowOffsetA / strideA] = 0;
            A_bytes[2][innerRowOffsetA / strideA] = 0;
            A_bytes[3][innerRowOffsetA / strideA] = 0;
          }
        }
        
        // Load next A scale
        if (threadIdx.x < (BN / VECTOR_SIZE)) {
          A_scale_temp = *reinterpret_cast<const float4 *>(
              &A_scale[(tileOffset / BLOCK_DIM) * N +
                       (colOffsetA + threadIdx.x * VECTOR_SIZE)]);
        }
        
        // Load next B tile
        for (uint16_t innerRowOffsetB = 0; innerRowOffsetB < BK;
             innerRowOffsetB += strideB) {
          if ((tileOffset + innerRowOffsetB + innerRowB) < K &&
              (colOffsetB + innerColB) < M &&
              (innerRowOffsetB + innerRowB) < BK) {
            // Store bytes for efficient packing later
            B_bytes[0][innerRowOffsetB / strideB] =
                *reinterpret_cast<const uint8_t *>(
                    &B[(tileOffset + innerRowOffsetB + innerRowB) * M +
                       (colOffsetB + innerColB)]);
            B_bytes[1][innerRowOffsetB / strideB] =
                *reinterpret_cast<const uint8_t *>(
                    &B[(tileOffset + innerRowOffsetB + innerRowB + 1) * M +
                       (colOffsetB + innerColB)]);
            B_bytes[2][innerRowOffsetB / strideB] =
                *reinterpret_cast<const uint8_t *>(
                    &B[(tileOffset + innerRowOffsetB + innerRowB + 2) * M +
                       (colOffsetB + innerColB)]);
            B_bytes[3][innerRowOffsetB / strideB] =
                *reinterpret_cast<const uint8_t *>(
                    &B[(tileOffset + innerRowOffsetB + innerRowB + 3) * M +
                       (colOffsetB + innerColB)]);
          } else if ((innerRowOffsetB + innerRowB) < BK) {
            B_bytes[0][innerRowOffsetB / strideB] = 0;
            B_bytes[1][innerRowOffsetB / strideB] = 0;
            B_bytes[2][innerRowOffsetB / strideB] = 0;
            B_bytes[3][innerRowOffsetB / strideB] = 0;
          }
        }
        
        // Load next B scale
        if (threadIdx.x == numThreads - 1) {
          B_scale_temp = B_scale[(tileOffset / BLOCK_DIM) * M_scale +
                               (colOffsetB / BLOCK_DIM)];
        }
      }

      // Compute current block
      long a[WITERN], b[WITERM];
      float b_scale = Ws[curr][BN];
      floatx16 c[WITERN][WITERM] = {0};
      
      // Process BK in chunks of 16 (matrix multiplication using MFMA)
      // Add pragmas for loop unrolling to improve instruction scheduling
      #pragma unroll 4
      for (uint16_t BKOffset = 0; BKOffset < BK; BKOffset += 16) {
        // Load A matrix elements
        #pragma unroll
        for (uint16_t wn = 0; wn < WITERN; ++wn) {
          a[wn] = *reinterpret_cast<long *>(
              &As[curr][warpRowOffset + wn * 32 + warpX][BKOffset + warpY * 8]);
        }
        
        // Load B matrix elements
        #pragma unroll
        for (uint16_t wm = 0; wm < WITERM; ++wm) {
          b[wm] = *reinterpret_cast<long *>(
              &Bs[curr][warpColOffset + wm * 32 + warpX][BKOffset + warpY * 8]);
        }
        
        // Matrix multiply using AMD MFMA instruction
        #pragma unroll
        for (uint16_t wn = 0; wn < WITERN; ++wn) {
          #pragma unroll
          for (uint16_t wm = 0; wm < WITERM; ++wm) {
            c[wn][wm] = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
                a[wn], b[wm], c[wn][wm], 0, 0, 0);
          }
        }
      }
      
      // Scale results with unrolled loops for better instruction scheduling
      #pragma unroll
      for (uint16_t wn = 0; wn < WITERN; ++wn) {
        #pragma unroll
        for (uint16_t wm = 0; wm < WITERM; ++wm) {
          #pragma unroll
          for (uint16_t j = 0; j < 4; ++j) {
            #pragma unroll
            for (uint16_t i = 0; i < 4; ++i) {
              d[wn][wm][i + j * 4] +=
                  c[wn][wm][i + j * 4] *
                  Ws[curr][warpRowOffset + wn * 32 + j * 8 + warpY * 4 + i] *
                  b_scale;
            }
          }
        }
      }

      // Store loaded data to shared memory (for next iteration)
      if (tileOffset < K) {
        // Store A with optimized packing
        for (uint16_t innerRowOffsetA = 0; innerRowOffsetA < BK;
             innerRowOffsetA += strideA) {
          if ((innerRowOffsetA + innerRowA) < BK) {
            // Optimized packing
            unsigned int pack = ((unsigned int)A_bytes[3][innerRowOffsetA / strideA] << 24) | 
                               ((unsigned int)A_bytes[2][innerRowOffsetA / strideA] << 16) | 
                               ((unsigned int)A_bytes[1][innerRowOffsetA / strideA] << 8) | 
                                (unsigned int)A_bytes[0][innerRowOffsetA / strideA];

            *reinterpret_cast<unsigned int *>(
                &As[next][innerColA][innerRowOffsetA + innerRowA]) = pack;
          }
        }
        
        // Store A scale
        if (threadIdx.x < (BN / VECTOR_SIZE)) {
          *reinterpret_cast<float4 *>(&Ws[next][threadIdx.x * VECTOR_SIZE]) = A_scale_temp;
        }
        
        // Store B with optimized packing
        for (uint16_t innerRowOffsetB = 0; innerRowOffsetB < BK;
             innerRowOffsetB += strideB) {
          if ((innerRowOffsetB + innerRowB) < BK) {
            // Optimized packing
            unsigned int pack = ((unsigned int)B_bytes[3][innerRowOffsetB / strideB] << 24) | 
                               ((unsigned int)B_bytes[2][innerRowOffsetB / strideB] << 16) | 
                               ((unsigned int)B_bytes[1][innerRowOffsetB / strideB] << 8) | 
                                (unsigned int)B_bytes[0][innerRowOffsetB / strideB];

            *reinterpret_cast<unsigned int *>(
                &Bs[next][innerColB][innerRowOffsetB + innerRowB]) = pack;
          }
        }
        
        // Store B scale
        if (threadIdx.x == numThreads - 1) {
          Ws[next][BN] = B_scale_temp;
        }
      }

      __syncthreads();

      // Toggle buffer indices
      curr = 1 - curr;
      next = 1 - next;
    }

    // Write final results to global memory with unrolled loops
    #pragma unroll
    for (uint16_t wn = 0; wn < WITERN; ++wn) {
      #pragma unroll
      for (uint16_t wm = 0; wm < WITERM; ++wm) {
        #pragma unroll
        for (uint16_t j = 0; j < 4; ++j) {
          #pragma unroll
          for (uint16_t i = 0; i < 4; ++i) {
            uint32_t globalRow = rowOffsetC + warpRowOffset + wn * 32 + j * 8 + warpY * 4 + i;
            uint32_t globalCol = colOffsetC + warpColOffset + wm * 32 + warpX;
            if (globalRow < N && globalCol < M) {
              C[globalRow * M + globalCol] = (__hip_bfloat16)d[wn][wm][i + j * 4];
            }
          }
        }
      }
    }
  }
}

// Optimized version matching original interface
torch::Tensor fp8_mm(torch::Tensor A, torch::Tensor B, torch::Tensor A_scale,
                     torch::Tensor B_scale, torch::Tensor C) {
  int N = A.size(0), K = A.size(1), M = B.size(0);

  const int BK = 64;
  const int BN = 128;
  const int BM = 128;
  const int WITERN = 1;
  const int WITERM = 1;
  const int SM_COUNT = 304;
  dim3 numThreads((BN * BM) / (16 * WITERN * WITERM));
  dim3 numBlocks(SM_COUNT);
  
  // Choose kernel based on matrix dimensions
  if (N > M) {
      fp8_mm_kernel<BN, BK, BM, WITERN, WITERM, SM_COUNT, TileIndexingStrategy::N_MAJOR><<<numBlocks, numThreads>>>(
          (__hip_fp8_e4m3_fnuz *)A.data_ptr(), (__hip_fp8_e4m3_fnuz *)B.data_ptr(),
          A_scale.data_ptr<float>(), B_scale.data_ptr<float>(),
          (__hip_bfloat16 *)C.data_ptr(), N, K, M);
  } else {
      fp8_mm_kernel<BN, BK, BM, WITERN, WITERM, SM_COUNT, TileIndexingStrategy::M_MAJOR><<<numBlocks, numThreads>>>(
          (__hip_fp8_e4m3_fnuz *)A.data_ptr(), (__hip_fp8_e4m3_fnuz *)B.data_ptr(),
          A_scale.data_ptr<float>(), B_scale.data_ptr<float>(),
          (__hip_bfloat16 *)C.data_ptr(), N, K, M);
  }
  return C;
}
"""

cpp_src = r"""
torch::Tensor fp8_mm(torch::Tensor A, torch::Tensor B, torch::Tensor A_scale,
                     torch::Tensor B_scale, torch::Tensor C);
"""

import sys
if sys.stdout is None:
    sys.stdout = open("/dev/stdout", "w")
if sys.stderr is None:
    sys.stderr = open("/dev/stderr", "w")

# Cache the compiled module to avoid recompilation
_cached_module = None

def get_module():
    global _cached_module
    if _cached_module is None:
        _cached_module = load_inline(
            name="fp8_mm",
            cpp_sources=[cpp_src],
            cuda_sources=[cuda_src],
            functions=["fp8_mm"],
            verbose=False,  # Set to False to reduce output
            extra_cuda_cflags=[
                "-O3",
                "--offload-arch=gfx942",
                "-std=c++20",
                "-Wno-unused-result",
                "-ffast-math",
                "-finline-functions",
                "-funroll-loops",
            ],
            with_cuda=True
        )
    return _cached_module

# Pre-compile the module on import
module = get_module()

# Cache the function reference to avoid attribute lookup
_fp8_mm_func = module.fp8_mm

def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    _fp8_mm_func(a, b, a_scale, b_scale, c)
    return c
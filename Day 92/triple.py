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

// Generic enum for tile index calculation strategy
enum class TileIndexingStrategy {
  M_MAJOR, // tileIdx / cdiv(M, BM) gives row, tileIdx % cdiv(M, BM) gives col
  N_MAJOR  // tileIdx % cdiv(N, BN) gives row, tileIdx / cdiv(N, BN) gives col
};

// Helper function for prefetching
template <typename T>
__device__ __forceinline__ void prefetch_global(const T* addr) {
    asm volatile("global_load_dwordx4 v[0:3], %0, off\n\t" :: "v"(addr));
}

template <const uint32_t BN, const uint32_t BK, const uint32_t BM,
          const uint32_t WITERN, const uint32_t WITERM, const uint32_t SM_COUNT,
          const TileIndexingStrategy strategy>
__global__
__attribute__((amdgpu_flat_work_group_size(0,0)))
__launch_bounds__(1024, 4)
void  fp8_mm_kernel(const __hip_fp8_e4m3_fnuz *A, const __hip_fp8_e4m3_fnuz *B,
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
  uint32_t rowOffsetC, colOffsetC;

  // Dynamic 2D grid distribution
  for (uint32_t tileIdx = blockIdx.x; tileIdx < numTiles; tileIdx += gridDim.x) {
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

    // Triple-buffering setup: three sets of shared memory buffers for better overlapping
    __shared__ __hip_fp8_e4m3_fnuz As1[BK][BN+8], Bs1[BK][BM+8];
    __shared__ __hip_fp8_e4m3_fnuz As2[BK][BN+8], Bs2[BK][BM+8];
    __shared__ __hip_fp8_e4m3_fnuz As3[BK][BN+8], Bs3[BK][BM+8];
    
    auto As = As1, Bs = Bs1;    // current buffer
    auto Ast = As2, Bst = Bs2;  // next buffer
    auto Astt = As3, Bstt = Bs3; // future buffer
    
    __shared__ float Ws1[BN + 1];
    __shared__ float Ws2[BN + 1];
    __shared__ float Ws3[BN + 1];
    auto Ws = Ws1;
    auto Wst = Ws2;
    auto Wstt = Ws3;

    __hip_fp8_e4m3_fnuz a[WITERN][8], b[WITERM][8];
    floatx16 d[WITERN][WITERM] = {0};

    // Prefetch addresses for the first tiles
    const __hip_fp8_e4m3_fnuz* A_ptr = nullptr;
    const __hip_fp8_e4m3_fnuz* B_ptr = nullptr;
    const float* A_scale_ptr = nullptr;
    const float* B_scale_ptr = nullptr;
    
    // Initial load: global memory -> shared memory
    for (uint32_t innerRowOffsetA = 0; innerRowOffsetA < BK;
         innerRowOffsetA += strideA) {
      if ((innerRowOffsetA + innerRowA) < K &&
          (colOffsetA + innerColA * VECTOR_SIZE) < N &&
          (innerRowOffsetA + innerRowA) < BK) {
        A_ptr = &A[(innerRowOffsetA + innerRowA) * N +
                   (colOffsetA + innerColA * VECTOR_SIZE)];
        prefetch_global(A_ptr);
        *reinterpret_cast<float *>(
            &As[innerRowOffsetA + innerRowA][innerColA * VECTOR_SIZE]) =
            *reinterpret_cast<const float *>(A_ptr);
      } else if ((innerRowOffsetA + innerRowA) < BK) {
        *reinterpret_cast<float *>(
            &As[innerRowOffsetA + innerRowA][innerColA * VECTOR_SIZE]) = 0.0f;
      }
    }
    
    if (threadIdx.x < SUBBN) {
      A_scale_ptr = &A_scale[(colOffsetA + threadIdx.x * VECTOR_SIZE)];
      prefetch_global(A_scale_ptr);
      *reinterpret_cast<float4 *>(&Ws[threadIdx.x * VECTOR_SIZE]) =
          *reinterpret_cast<const float4 *>(A_scale_ptr);
    }
    
    for (uint32_t innerRowOffsetB = 0; innerRowOffsetB < BK;
         innerRowOffsetB += strideB) {
      if ((innerRowOffsetB + innerRowB) < K &&
          (colOffsetB + innerColB * VECTOR_SIZE) < M &&
          (innerRowOffsetB + innerRowB) < BK) {
        B_ptr = &B[(innerRowOffsetB + innerRowB) * M +
                   (colOffsetB + innerColB * VECTOR_SIZE)];
        prefetch_global(B_ptr);
        *reinterpret_cast<float *>(
            &Bs[innerRowOffsetB + innerRowB][innerColB * VECTOR_SIZE]) =
            *reinterpret_cast<const float *>(B_ptr);
      } else if ((innerRowOffsetB + innerRowB) < BK &&
                 (innerColB * VECTOR_SIZE) < BM) {
        *reinterpret_cast<float *>(
            &Bs[innerRowOffsetB + innerRowB][innerColB * VECTOR_SIZE]) = 0.0f;
      }
    }
    
    if (threadIdx.x == numThreads - 1) {
      B_scale_ptr = &B_scale[(colOffsetB / BLOCK_DIM)];
      prefetch_global(B_scale_ptr);
      Ws[BN] = *B_scale_ptr;
    }

    __syncthreads();

    // Prefetch addresses for the next tile
    const __hip_fp8_e4m3_fnuz* next_A_ptr = nullptr;
    const __hip_fp8_e4m3_fnuz* next_B_ptr = nullptr;
    const float* next_A_scale_ptr = nullptr;
    const float* next_B_scale_ptr = nullptr;
    
    if (BK < K) {
      // Prefetch data for next tile
      for (uint32_t innerRowOffsetA = 0; innerRowOffsetA < BK;
           innerRowOffsetA += strideA) {
        if ((BK + innerRowOffsetA + innerRowA) < K &&
            (colOffsetA + innerColA * VECTOR_SIZE) < N &&
            (innerRowOffsetA + innerRowA) < BK) {
          next_A_ptr = &A[(BK + innerRowOffsetA + innerRowA) * N +
                        (colOffsetA + innerColA * VECTOR_SIZE)];
          prefetch_global(next_A_ptr);
        }
      }
      
      if (threadIdx.x < SUBBN) {
        next_A_scale_ptr = &A_scale[(BK / BLOCK_DIM) * N +
                                 (colOffsetA + threadIdx.x * VECTOR_SIZE)];
        prefetch_global(next_A_scale_ptr);
      }
      
      for (uint32_t innerRowOffsetB = 0; innerRowOffsetB < BK;
           innerRowOffsetB += strideB) {
        if ((BK + innerRowOffsetB + innerRowB) < K &&
            (colOffsetB + innerColB * VECTOR_SIZE) < M &&
            (innerRowOffsetB + innerRowB) < BK) {
          next_B_ptr = &B[(BK + innerRowOffsetB + innerRowB) * M +
                        (colOffsetB + innerColB * VECTOR_SIZE)];
          prefetch_global(next_B_ptr);
        }
      }
      
      if (threadIdx.x == numThreads - 1) {
        next_B_scale_ptr = &B_scale[(BK / BLOCK_DIM) * M_scale +
                                  (colOffsetB / BLOCK_DIM)];
        prefetch_global(next_B_scale_ptr);
      }
    }

    // Triple buffering for better overlapping of computation and memory access
    float Adata[4], Bdata[4];
    float4 Asdata;
    float Bsdata;
    
    // Main computation loop with triple buffering
    for (uint32_t tileOffset = 0; tileOffset < K; tileOffset += BK) {
      // Compute current block
      float b_scale = Ws[BN];
      floatx16 c[WITERN][WITERM] = {0};
      
      // Load data for the future tile (tileOffset + 2*BK) if needed
      if (tileOffset + 2*BK < K) {
        // Load future A tile
        for (uint32_t innerRowOffsetA = 0; innerRowOffsetA < BK;
             innerRowOffsetA += strideA) {
          if ((tileOffset + 2*BK + innerRowOffsetA + innerRowA) < K &&
              (colOffsetA + innerColA * VECTOR_SIZE) < N &&
              (innerRowOffsetA + innerRowA) < BK) {
            A_ptr = &A[(tileOffset + 2*BK + innerRowOffsetA + innerRowA) * N +
                       (colOffsetA + innerColA * VECTOR_SIZE)];
            prefetch_global(A_ptr);
            Adata[innerRowOffsetA / strideA] = *reinterpret_cast<const float *>(A_ptr);
          } else if ((innerRowOffsetA + innerRowA) < BK) {
            Adata[innerRowOffsetA / strideA] = 0.0f;
          }
        }
        
        // Load future A scale
        if (threadIdx.x < SUBBN) {
          A_scale_ptr = &A_scale[((tileOffset + 2*BK) / BLOCK_DIM) * N +
                               (colOffsetA + threadIdx.x * VECTOR_SIZE)];
          prefetch_global(A_scale_ptr);
          Asdata = *reinterpret_cast<const float4 *>(A_scale_ptr);
        }
        
        // Load future B tile
        for (uint32_t innerRowOffsetB = 0; innerRowOffsetB < BK;
             innerRowOffsetB += strideB) {
          if ((tileOffset + 2*BK + innerRowOffsetB + innerRowB) < K &&
              (colOffsetB + innerColB * VECTOR_SIZE) < M &&
              (innerRowOffsetB + innerRowB) < BK) {
            B_ptr = &B[(tileOffset + 2*BK + innerRowOffsetB + innerRowB) * M +
                       (colOffsetB + innerColB * VECTOR_SIZE)];
            prefetch_global(B_ptr);
            Bdata[innerRowOffsetB / strideB] = *reinterpret_cast<const float *>(B_ptr);
          } else if ((innerRowOffsetB + innerRowB) < BK &&
                     (innerColB * VECTOR_SIZE) < BM) {
            Bdata[innerRowOffsetB / strideB] = 0.0f;
          }
        }
        
        // Load future B scale
        if (threadIdx.x == numThreads - 1) {
          B_scale_ptr = &B_scale[((tileOffset + 2*BK) / BLOCK_DIM) * M_scale +
                               (colOffsetB / BLOCK_DIM)];
          prefetch_global(B_scale_ptr);
          Bsdata = *B_scale_ptr;
        }
        
        // Store future data to shared memory
        for (uint32_t innerRowOffsetA = 0; innerRowOffsetA < BK;
             innerRowOffsetA += strideA) {
          if ((innerRowOffsetA + innerRowA) < BK) {
            *reinterpret_cast<float *>(
                &Astt[innerRowOffsetA + innerRowA][innerColA * VECTOR_SIZE]) =
                Adata[innerRowOffsetA / strideA];
          }
        }
        
        if (threadIdx.x < SUBBN) {
          *reinterpret_cast<float4 *>(&Wstt[threadIdx.x * VECTOR_SIZE]) = Asdata;
        }
        
        for (uint32_t innerRowOffsetB = 0; innerRowOffsetB < BK;
             innerRowOffsetB += strideB) {
          if ((innerRowOffsetB + innerRowB) < BK &&
              (innerColB * VECTOR_SIZE) < BM) {
            *reinterpret_cast<float *>(
                &Bstt[innerRowOffsetB + innerRowB][innerColB * VECTOR_SIZE]) =
                Bdata[innerRowOffsetB / strideB];
          }
        }
        
        if (threadIdx.x == numThreads - 1) {
          Wstt[BN] = Bsdata;
        }
      }
      
      // Process BK in chunks of 16 (matrix multiplication using MFMA)
      #pragma unroll 4
      for (uint32_t BKOffset = 0; BKOffset < BK; BKOffset += 16) {
        // Load A matrix elements
        #pragma unroll
        for (uint32_t wn = 0; wn < WITERN; ++wn) {
          #pragma unroll
          for (uint32_t i = 0; i < 8; ++i) {
            a[wn][i] =
                As[BKOffset + warpY * 8 + i][warpRowOffset + wn * 32 + warpX];
          }
        }
        
        // Load B matrix elements
        #pragma unroll
        for (uint32_t wm = 0; wm < WITERM; ++wm) {
          #pragma unroll
          for (uint32_t i = 0; i < 8; ++i) {
            b[wm][i] =
                Bs[BKOffset + warpY * 8 + i][warpColOffset + wm * 32 + warpX];
          }
        }
        
        // Matrix multiply using AMD MFMA instruction
        #pragma unroll
        for (uint32_t wn = 0; wn < WITERN; ++wn) {
          #pragma unroll
          for (uint32_t wm = 0; wm < WITERM; ++wm) {
            c[wn][wm] = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
                *reinterpret_cast<long *>(a[wn]),
                *reinterpret_cast<long *>(b[wm]), c[wn][wm], 0, 0, 0);
          }
        }
      }
      
      // Scale results
      #pragma unroll
      for (uint32_t wn = 0; wn < WITERN; ++wn) {
        #pragma unroll
        for (uint32_t wm = 0; wm < WITERM; ++wm) {
          #pragma unroll
          for (uint32_t j = 0; j < 4; ++j) {
            #pragma unroll 
            for (uint32_t i = 0; i < 4; ++i) {
              d[wn][wm][i + j * 4] +=
                  c[wn][wm][i + j * 4] *
                  Ws[warpRowOffset + wn * 32 + j * 8 + warpY * 4 + i] * b_scale;
            }
          }
        }
      }

      // Swap buffer pointers for next iteration - triple buffering
      auto tmp = As;
      As = Ast;
      Ast = Astt;
      Astt = tmp;
      
      tmp = Bs;
      Bs = Bst;
      Bst = Bstt;
      Bstt = tmp;
      
      auto tmp2 = Ws;
      Ws = Wst;
      Wst = Wstt;
      Wstt = tmp2;

      __syncthreads();
      
      // Prefetch the next set of data if we're not at the end
      if (tileOffset + 3*BK < K) {
        // Prefetch addresses
        for (uint32_t innerRowOffsetA = 0; innerRowOffsetA < BK;
             innerRowOffsetA += strideA) {
          if ((tileOffset + 3*BK + innerRowOffsetA + innerRowA) < K &&
              (colOffsetA + innerColA * VECTOR_SIZE) < N) {
            prefetch_global(&A[(tileOffset + 3*BK + innerRowOffsetA + innerRowA) * N +
                           (colOffsetA + innerColA * VECTOR_SIZE)]);
          }
        }
        
        if (threadIdx.x < SUBBN) {
          prefetch_global(&A_scale[((tileOffset + 3*BK) / BLOCK_DIM) * N +
                                 (colOffsetA + threadIdx.x * VECTOR_SIZE)]);
        }
        
        for (uint32_t innerRowOffsetB = 0; innerRowOffsetB < BK;
             innerRowOffsetB += strideB) {
          if ((tileOffset + 3*BK + innerRowOffsetB + innerRowB) < K &&
              (colOffsetB + innerColB * VECTOR_SIZE) < M) {
            prefetch_global(&B[(tileOffset + 3*BK + innerRowOffsetB + innerRowB) * M +
                           (colOffsetB + innerColB * VECTOR_SIZE)]);
          }
        }
        
        if (threadIdx.x == numThreads - 1) {
          prefetch_global(&B_scale[((tileOffset + 3*BK) / BLOCK_DIM) * M_scale +
                                 (colOffsetB / BLOCK_DIM)]);
        }
      }
    }

    // Write final results to global memory using vectorized stores
    #pragma unroll
    for (uint32_t wn = 0; wn < WITERN; ++wn) {
      #pragma unroll
      for (uint32_t wm = 0; wm < WITERM; ++wm) {
        // Process outputs in groups of 4 for SIMD efficiency
        for (uint32_t j = 0; j < 4; ++j) {
          // Convert 4 float values to bf16 in one operation when possible
          uint32_t globalRow = rowOffsetC + warpRowOffset + wn * 32 + j * 8 + warpY * 4;
          uint32_t globalCol = colOffsetC + warpColOffset + wm * 32 + warpX;
          
          if (globalRow < N && globalCol < M) {
            #pragma unroll
            for (uint32_t i = 0; i < 4; ++i) {
              if (globalRow + i < N && globalCol < M) {
                C[(globalRow + i) * M + globalCol] = (__hip_bfloat16)d[wn][wm][i + j * 4];
              }
            }
          }
        }
      }
    }
  }
}

// Function to select optimal kernel parameters based on matrix dimensions
template <TileIndexingStrategy strategy>
void launch_optimized_kernel(
    const __hip_fp8_e4m3_fnuz *A, const __hip_fp8_e4m3_fnuz *B,
    const float *A_scale, const float *B_scale, __hip_bfloat16 *C,
    int N, int K, int M, dim3 numBlocks) {
    
    constexpr int SM_COUNT = 304;  // MI300X has 304 CUs
    
    // Optimize parameters based on matrix size
    if (K >= 4096) {
        // Large K dimension - use larger BK for better reuse
        constexpr int BK = 128;
        constexpr int BN = 128;
        constexpr int BM = 128;
        constexpr int WITERN = 1;  // Use more wavefronts per block
        constexpr int WITERM = 1;
        dim3 numThreads((BN * BM) / (16 * WITERN * WITERM));
        
        fp8_mm_kernel<BN, BK, BM, WITERN, WITERM, SM_COUNT, strategy><<<numBlocks, numThreads>>>(
            A, B, A_scale, B_scale, C, N, K, M);
    } 
    else if (K < 512) {
        // Small K dimension - optimize for wider tiles
        constexpr int BK = 64; 
        constexpr int BN = 256;
        constexpr int BM = 128;
        constexpr int WITERN = 1;
        constexpr int WITERM = 1;
        dim3 numThreads((BN * BM) / (16 * WITERN * WITERM));
        
        fp8_mm_kernel<BN, BK, BM, WITERN, WITERM, SM_COUNT, strategy><<<numBlocks, numThreads>>>(
            A, B, A_scale, B_scale, C, N, K, M);
    }
    else {
        // Medium K - balanced parameters
        constexpr int BK = 64;
        constexpr int BN = 128;
        constexpr int BM = 128;
        constexpr int WITERN = 1;
        constexpr int WITERM = 1;
        dim3 numThreads((BN * BM) / (16 * WITERN * WITERM));
        
        fp8_mm_kernel<BN, BK, BM, WITERN, WITERM, SM_COUNT, strategy><<<numBlocks, numThreads>>>(
            A, B, A_scale, B_scale, C, N, K, M);
    }
}

at::Tensor fp8_mm(at::Tensor A, at::Tensor B, at::Tensor A_scale,
                  at::Tensor B_scale, at::Tensor C) {
  int N = A.size(0), K = A.size(1), M = B.size(0);
  
  // Calculate optimal number of blocks
  const int SM_COUNT = 304;  // MI300X has 304 CUs
  int numTiles = cdiv(N, 128) * cdiv(M, 128);
  dim3 numBlocks(std::min(numTiles, SM_COUNT * 4)); // Use multiple blocks per SM for better occupancy
  
  // Choose kernel based on matrix dimensions
  if (N > M) {
      launch_optimized_kernel<TileIndexingStrategy::N_MAJOR>(
          (__hip_fp8_e4m3_fnuz *)A.data_ptr(), (__hip_fp8_e4m3_fnuz *)B.data_ptr(),
          A_scale.data_ptr<float>(), B_scale.data_ptr<float>(),
          (__hip_bfloat16 *)C.data_ptr(), N, K, M, numBlocks);
  } else {
      launch_optimized_kernel<TileIndexingStrategy::M_MAJOR>(
          (__hip_fp8_e4m3_fnuz *)A.data_ptr(), (__hip_fp8_e4m3_fnuz *)B.data_ptr(),
          A_scale.data_ptr<float>(), B_scale.data_ptr<float>(),
          (__hip_bfloat16 *)C.data_ptr(), N, K, M, numBlocks);
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
        "-ffast-math",
        "-mfma",
    ],
)

def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    return module.fp8_mm(a, b, a_scale, b_scale, c)
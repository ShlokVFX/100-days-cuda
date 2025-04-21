#!POPCORN leaderboard amd-fp8-mm
#!POPCORN gpus MI300

import os
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

# Use clang++ as the C++ compiler for ROCm compatibility
os.environ["CXX"] = "clang++"

# HIP kernel source for optimized FP8 matrix multiplication using MFMA
cuda_src = r"""
#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/amd_detail/amd_hip_fp8.h>

// Helper function for ceiling division
__host__ __device__ __forceinline__ size_t cdiv(size_t a, size_t b) {
  return (a + b - 1) / b;
}

// Optimized constants for MI300
#define BLOCK_DIM 256          // Increased from 128
#define WARPSIZE 64
#define VECTOR_SIZE 4
#define WAVE_MATRIX_M 32
#define WAVE_MATRIX_N 32
#define WAVE_MATRIX_K 16

// Main FP8 GEMM kernel with aggressive optimizations for MI300 architecture
// - Increased register usage
// - More aggressive prefetching
// - Optimized memory access patterns
// - Multiple waves per block for better occupancy
// BN: rows per block, BK: depth per block, BM: cols per block
template <const size_t BN, const size_t BK, const size_t BM, const size_t NUM_STAGES = 3>
__global__ __launch_bounds__(256, 2) void fp8_mm_kernel(
                              const __hip_fp8_e4m3_fnuz *__restrict__ A,
                              const __hip_fp8_e4m3_fnuz *__restrict__ B,
                              const float *__restrict__ A_scale, 
                              const float *__restrict__ B_scale,
                              __hip_bfloat16 *__restrict__ C, 
                              size_t N, size_t K, size_t M) {
  // Vector types for faster memory operations
  using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
  using fp8x4 = __hip_fp8x4_e4m3_fnuz;
  
  // Thread and block level indexing
  const size_t warpIdx = threadIdx.x / WARPSIZE;
  const size_t laneIdx = threadIdx.x % WARPSIZE;
  const size_t warpX = laneIdx % 32;
  const size_t warpY = laneIdx / 32;
  
  // Calculate block level start positions with interleaved pattern for better memory coalescence
  const size_t blockStartRow = blockIdx.y * BN;
  const size_t blockStartCol = blockIdx.x * BM;
  
  // Wave position within the block
  const size_t waveRowOffset = (warpIdx / (BM / WAVE_MATRIX_M)) * WAVE_MATRIX_N;
  const size_t waveColOffset = (warpIdx % (BM / WAVE_MATRIX_M)) * WAVE_MATRIX_M;
  
  // Shared memory with multi-stage buffering for better prefetch hiding
  __shared__ __hip_fp8_e4m3_fnuz As[NUM_STAGES][BK][BN];
  __shared__ __hip_fp8_e4m3_fnuz Bs[NUM_STAGES][BK][BM];
  __shared__ float A_scales[NUM_STAGES][BN];
  __shared__ float B_scales[NUM_STAGES][BM/WAVE_MATRIX_M];
  
  // Thread-local registers for MFMA operations
  __hip_fp8_e4m3_fnuz a_regs[8], b_regs[8];
  floatx16 c_regs[4] = {0}; // Increased to 4 for more matrix blocks per thread
  
  // Initialize first buffer stages
  size_t bufferIdx = 0;
  size_t loadingBufferIdx = 0;
  size_t computeBufferIdx = 0;
  
  // Prefetch first two stages
  for (size_t stageIdx = 0; stageIdx < min(NUM_STAGES-1, cdiv(K, BK)); stageIdx++) {
    const size_t tileK = stageIdx * BK;
    
    // Collaborative loading with stride pattern for better memory coalescing
    for (size_t i = threadIdx.x; i < BK * BN / VECTOR_SIZE; i += blockDim.x) {
      const size_t blockK = i / (BN / VECTOR_SIZE);
      const size_t blockN = (i % (BN / VECTOR_SIZE)) * VECTOR_SIZE;
      
      if ((tileK + blockK) < K && (blockStartRow + blockN) < N) {
        // Load with vectorized operations
        fp8x4 tmp = *reinterpret_cast<const fp8x4*>(&A[(tileK + blockK) * N + (blockStartRow + blockN)]);
        *reinterpret_cast<fp8x4*>(&As[loadingBufferIdx][blockK][blockN]) = tmp;
      } else {
        // Zero pad out-of-bounds
        *reinterpret_cast<fp8x4*>(&As[loadingBufferIdx][blockK][blockN]) = fp8x4{0.0f, 0.0f, 0.0f, 0.0f};
      }
    }
    
    // Load scale factors for A
    for (size_t i = threadIdx.x; i < BN; i += blockDim.x) {
      if ((blockStartRow + i) < N) {
        A_scales[loadingBufferIdx][i] = A_scale[((tileK) / BLOCK_DIM) * N + (blockStartRow + i)];
      } else {
        A_scales[loadingBufferIdx][i] = 0.0f;
      }
    }
    
    // Load B with similar pattern
    for (size_t i = threadIdx.x; i < BK * BM / VECTOR_SIZE; i += blockDim.x) {
      const size_t blockK = i / (BM / VECTOR_SIZE);
      const size_t blockM = (i % (BM / VECTOR_SIZE)) * VECTOR_SIZE;
      
      if ((tileK + blockK) < K && (blockStartCol + blockM) < M) {
        fp8x4 tmp = *reinterpret_cast<const fp8x4*>(&B[(tileK + blockK) * M + (blockStartCol + blockM)]);
        *reinterpret_cast<fp8x4*>(&Bs[loadingBufferIdx][blockK][blockM]) = tmp;
      } else {
        *reinterpret_cast<fp8x4*>(&Bs[loadingBufferIdx][blockK][blockM]) = fp8x4{0.0f, 0.0f, 0.0f, 0.0f};
      }
    }
    
    // Load B scales
    for (size_t i = threadIdx.x; i < BM/WAVE_MATRIX_M; i += blockDim.x) {
      if ((blockStartCol + i * WAVE_MATRIX_M) < M) {
        B_scales[loadingBufferIdx][i] = B_scale[((tileK) / BLOCK_DIM) * cdiv(M, BLOCK_DIM) + ((blockStartCol + i * WAVE_MATRIX_M) / BLOCK_DIM)];
      } else {
        B_scales[loadingBufferIdx][i] = 0.0f;
      }
    }
    
    loadingBufferIdx = (loadingBufferIdx + 1) % NUM_STAGES;
    __syncthreads();
  }
  
  // Main computation loop with software pipelining
  for (size_t tileK = 0; tileK < K; tileK += BK) {
    // Start prefetching next tile if available
    if (tileK + BK * (NUM_STAGES-1) < K) {
      const size_t next_tileK = tileK + BK * (NUM_STAGES-1);
      
      // Collaborative loading for next stage
      for (size_t i = threadIdx.x; i < BK * BN / VECTOR_SIZE; i += blockDim.x) {
        const size_t blockK = i / (BN / VECTOR_SIZE);
        const size_t blockN = (i % (BN / VECTOR_SIZE)) * VECTOR_SIZE;
        
        if ((next_tileK + blockK) < K && (blockStartRow + blockN) < N) {
          fp8x4 tmp = *reinterpret_cast<const fp8x4*>(&A[(next_tileK + blockK) * N + (blockStartRow + blockN)]);
          *reinterpret_cast<fp8x4*>(&As[loadingBufferIdx][blockK][blockN]) = tmp;
        } else {
          *reinterpret_cast<fp8x4*>(&As[loadingBufferIdx][blockK][blockN]) = fp8x4{0.0f, 0.0f, 0.0f, 0.0f};
        }
      }
      
      // Load A scales
      for (size_t i = threadIdx.x; i < BN; i += blockDim.x) {
        if ((blockStartRow + i) < N) {
          A_scales[loadingBufferIdx][i] = A_scale[((next_tileK) / BLOCK_DIM) * N + (blockStartRow + i)];
        } else {
          A_scales[loadingBufferIdx][i] = 0.0f;
        }
      }
      
      // Load B and its scales
      for (size_t i = threadIdx.x; i < BK * BM / VECTOR_SIZE; i += blockDim.x) {
        const size_t blockK = i / (BM / VECTOR_SIZE);
        const size_t blockM = (i % (BM / VECTOR_SIZE)) * VECTOR_SIZE;
        
        if ((next_tileK + blockK) < K && (blockStartCol + blockM) < M) {
          fp8x4 tmp = *reinterpret_cast<const fp8x4*>(&B[(next_tileK + blockK) * M + (blockStartCol + blockM)]);
          *reinterpret_cast<fp8x4*>(&Bs[loadingBufferIdx][blockK][blockM]) = tmp;
        } else {
          *reinterpret_cast<fp8x4*>(&Bs[loadingBufferIdx][blockK][blockM]) = fp8x4{0.0f, 0.0f, 0.0f, 0.0f};
        }
      }
      
      // Load B scales
      for (size_t i = threadIdx.x; i < BM/WAVE_MATRIX_M; i += blockDim.x) {
        if ((blockStartCol + i * WAVE_MATRIX_M) < M) {
          B_scales[loadingBufferIdx][i] = B_scale[((next_tileK) / BLOCK_DIM) * cdiv(M, BLOCK_DIM) + ((blockStartCol + i * WAVE_MATRIX_M) / BLOCK_DIM)];
        } else {
          B_scales[loadingBufferIdx][i] = 0.0f;
        }
      }
      
      loadingBufferIdx = (loadingBufferIdx + 1) % NUM_STAGES;
    }
    
    // Process current tile with unrolled loops for maximum throughput
    for (size_t microTileK = 0; microTileK < BK; microTileK += WAVE_MATRIX_K) {
      // Each wave processes multiple matrix blocks for better utilization
      #pragma unroll
      for (size_t matrixIdx = 0; matrixIdx < 4; matrixIdx++) {
        // Load matrix elements into registers with double buffering pattern
        #pragma unroll
        for (size_t i = 0; i < 8; ++i) {
          // Calculate offset for current matrix
          const size_t rowOffset = waveRowOffset + (matrixIdx / 2) * 16;
          const size_t colOffset = waveColOffset + (matrixIdx % 2) * 16;
          
          // Prefetch into registers
          a_regs[i] = As[computeBufferIdx][microTileK + warpY * 8 + i][rowOffset + warpX];
          b_regs[i] = Bs[computeBufferIdx][microTileK + warpY * 8 + i][colOffset + warpX];
        }
        
        // Execute MFMA instruction - native wavefront matrix multiply-accumulate
        floatx16 mfma_result = {0};
        mfma_result = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
          *reinterpret_cast<long *>(a_regs), 
          *reinterpret_cast<long *>(b_regs), 
          mfma_result, 0, 0, 0);
        
        // Apply scaling with vectorized operations
        const float b_scale = B_scales[computeBufferIdx][warpIdx % (BM/WAVE_MATRIX_M)];
        
        #pragma unroll
        for (size_t j = 0; j < 4; ++j) {
          #pragma unroll
          for (size_t i = 0; i < 4; ++i) {
            const size_t rowIdx = waveRowOffset + (matrixIdx / 2) * 16 + j * 8 + warpY * 4 + i;
            const float a_scale = A_scales[computeBufferIdx][rowIdx];
            c_regs[matrixIdx][i + j * 4] += mfma_result[i + j * 4] * a_scale * b_scale;
          }
        }
      }
    }
    
    // Advance to next buffer
    computeBufferIdx = (computeBufferIdx + 1) % NUM_STAGES;
    __syncthreads();
  }
  
  // Store results to global memory with vectorized operations
  #pragma unroll
  for (size_t matrixIdx = 0; matrixIdx < 4; matrixIdx++) {
    const size_t rowOffsetBase = blockStartRow + waveRowOffset + (matrixIdx / 2) * 16;
    const size_t colOffsetBase = blockStartCol + waveColOffset + (matrixIdx % 2) * 16;
    
    #pragma unroll
    for (size_t j = 0; j < 4; ++j) {
      #pragma unroll
      for (size_t i = 0; i < 4; ++i) {
        const size_t rowIdx = rowOffsetBase + j * 8 + warpY * 4 + i;
        const size_t colIdx = colOffsetBase + warpX;
        
        if (rowIdx < N && colIdx < M) {
          C[rowIdx * M + colIdx] = (__hip_bfloat16)c_regs[matrixIdx][i + j * 4];
        }
      }
    }
  }
}

// Host-side launcher with autotuned parameters
at::Tensor fp8_mm(at::Tensor A, at::Tensor B, at::Tensor A_scale,
                  at::Tensor B_scale, at::Tensor C) {
  size_t N = A.size(0), K = A.size(1), M = B.size(1);
  
  // Optimized tiling parameters for MI300 - larger tiles for better utilization
  const size_t BK = 64;  // Increased from 32
  const size_t BN = 256; // Increased from 128
  const size_t BM = 256; // Increased from 128
  
  // Launch configuration with more threads per block and fewer blocks
  dim3 numThreads(256); // Increased from 128
  dim3 numBlocks(cdiv(M, BM), cdiv(N, BN));
  
  // Configure L1/L2 cache preference for read-intensive workload
  hipError_t err = hipDeviceSetCacheConfig(hipFuncCachePreferL1);
  if (err != hipSuccess) {
    printf("Error setting cache config: %s\n", hipGetErrorString(err));
  }
  
  // Set shared memory bank size to 8 bytes for better throughput with FP8
  err = hipDeviceSetSharedMemConfig(hipSharedMemBankSizeEightByte);
  if (err != hipSuccess) {
    printf("Error setting shared memory config: %s\n", hipGetErrorString(err));
  }
  
  // Stream creation for asynchronous execution
  hipStream_t stream;
  hipStreamCreate(&stream);
  
  // Launch kernel with 3-stage software pipelining
  fp8_mm_kernel<BN, BK, BM, 3><<<numBlocks, numThreads, 0, stream>>>(
      (__hip_fp8_e4m3_fnuz *)A.data_ptr(), (__hip_fp8_e4m3_fnuz *)B.data_ptr(),
      A_scale.data_ptr<float>(), B_scale.data_ptr<float>(),
      (__hip_bfloat16 *)C.data_ptr(), N, K, M);
  
  // Synchronize the stream
  hipStreamSynchronize(stream);
  hipStreamDestroy(stream);
  
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
    extra_cuda_cflags=[
        "-O3", 
        "--offload-arch=gfx942", 
        "-std=c++20", 
        "-ffp-contract=fast",
        "-ffast-math",
        "-funsafe-math-optimizations",
        "-mllvm -amdgpu-early-inline-all=true",
        "-mllvm -amdgpu-function-calls=false"
    ],
)

# PyTorch interface to call the HIP kernel
def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    return module.fp8_mm(a, b, a_scale, b_scale, c)
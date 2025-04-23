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

// Block sizes optimized for performance
#define BLOCK_N 64
#define BLOCK_K 32
#define BLOCK_M 128

// Vector size for memory accesses
#define VECTOR_SIZE 4

// Pre-launch kernel to warm up the GPU and initialize the driver
__global__ void warmup_kernel() {
  // Simple computation to ensure the kernel executes
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  float result = 0.0f;
  for (int i = 0; i < 100; i++) {
    result += sinf(idx + i);
  }
  // Prevent compiler from optimizing away the computation
  if (result == 0.0f) {
    __threadfence();
  }
}

template <const size_t BN, const size_t BK, const size_t BM>
__global__ void fp8_mm_kernel(const __hip_fp8_e4m3_fnuz *A,
                              const __hip_fp8_e4m3_fnuz *B,
                              const float *A_scale, const float *B_scale,
                              __hip_bfloat16 *C, size_t N, size_t K, size_t M) {
  using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

  static constexpr size_t WARPSIZE = 64;
  static constexpr size_t strideA = (BM / VECTOR_SIZE);
  static constexpr size_t strideB = (BN / VECTOR_SIZE);
  
  // Get row and column offsets for this thread block
  size_t rowOffsetC = blockIdx.y * BN;
  size_t colOffsetC = blockIdx.x * BM;
  size_t colOffsetA = rowOffsetC;
  size_t colOffsetB = colOffsetC;

  // Calculate thread indices within the block
  size_t laneIdx = threadIdx.x % WARPSIZE;
  size_t warpIdx = threadIdx.x / WARPSIZE;
  
  // Calculate warp offsets within the block
  size_t warpColOffset = (warpIdx % (BM / 32)) * 32;
  size_t warpRowOffset = (warpIdx / (BM / 32)) * 32;
  size_t warpX = laneIdx % 32;
  size_t warpY = laneIdx / 32;

  // Shared memory allocations for double buffering
  __shared__ __hip_fp8_e4m3_fnuz As[2][BK][BN];
  __shared__ __hip_fp8_e4m3_fnuz Bs[2][BK][BM];
  __shared__ float Ws[2][BN + 1];  // +1 for B scale

  // Registers for loading data
  __hip_fp8_e4m3_fnuz a[8], b[8];
  
  // Accumulator registers
  floatx16 d = {0};

  // Thread indices for vector loading
  size_t innerColA = threadIdx.x % (BN / VECTOR_SIZE);
  size_t innerRowA = threadIdx.x / (BN / VECTOR_SIZE);
  size_t innerColB = threadIdx.x % (BM / VECTOR_SIZE);
  size_t innerRowB = threadIdx.x / (BM / VECTOR_SIZE);
  
  // Double buffering indices
  int current_buffer = 0;
  int next_buffer = 1;

  // Load the first tile
  size_t tileOffset = 0;

  // Load A, A_scale, B, and B_scale for the first tile
  if (tileOffset < K) {
    // Load matrix A into shared memory
    for (size_t innerRowOffsetA = 0; innerRowOffsetA < BK; innerRowOffsetA += strideA) {
      if ((tileOffset + innerRowOffsetA + innerRowA) < K &&
          (colOffsetA + innerColA * VECTOR_SIZE) < N) {
        __hip_fp8x4_e4m3_fnuz tmp = *(__hip_fp8x4_e4m3_fnuz *)&A[(tileOffset + innerRowOffsetA + innerRowA) * N + (colOffsetA + innerColA * VECTOR_SIZE)];
        *(__hip_fp8x4_e4m3_fnuz *)&As[current_buffer][innerRowOffsetA + innerRowA][innerColA * VECTOR_SIZE] = tmp;

        // Load A scale
        if (threadIdx.x < (BN / VECTOR_SIZE)) {
          float4 tmp = *(float4 *)&A_scale[((tileOffset + innerRowOffsetA + innerRowA) / BLOCK_DIM) * N + 
                          (colOffsetA + innerColA * VECTOR_SIZE)];
          *(float4 *)&Ws[current_buffer][threadIdx.x * VECTOR_SIZE] = tmp;
        }
      } else {
        *(__hip_fp8x4_e4m3_fnuz *)&As[current_buffer][innerRowOffsetA + innerRowA][innerColA * VECTOR_SIZE] = float4{0.0f, 0.0f, 0.0f, 0.0f};
      }
    }

    // Load matrix B into shared memory
    for (size_t innerRowOffsetB = 0; innerRowOffsetB < BK; innerRowOffsetB += strideB) {
      if ((tileOffset + innerRowOffsetB + innerRowB) < K &&
          (colOffsetB + innerColB * VECTOR_SIZE) < M) {
        __hip_fp8x4_e4m3_fnuz tmp = *(__hip_fp8x4_e4m3_fnuz *)&B[(tileOffset + innerRowOffsetB + innerRowB) * M + (colOffsetB + innerColB * VECTOR_SIZE)];
        *(__hip_fp8x4_e4m3_fnuz *)&Bs[current_buffer][innerRowOffsetB + innerRowB][innerColB * VECTOR_SIZE] = tmp;

        // Load B scale
        if (threadIdx.x == 0) {
          Ws[current_buffer][BN] = B_scale[((tileOffset + innerRowOffsetB + innerRowB) / BLOCK_DIM) * cdiv(M, BLOCK_DIM) + 
                                         ((colOffsetB + innerColB * VECTOR_SIZE) / BLOCK_DIM)];
        }
      } else {
        *(__hip_fp8x4_e4m3_fnuz *)&Bs[current_buffer][innerRowOffsetB + innerRowB][innerColB * VECTOR_SIZE] = float4{0.0f, 0.0f, 0.0f, 0.0f};
      }
    }
  }

  __syncthreads();

  // Main computation loop with double buffering
  for (tileOffset = 0; tileOffset < K; tileOffset += BK) {
    // Prefetch next tile if available
    if (tileOffset + BK < K) {
      // Load next A tile
      for (size_t innerRowOffsetA = 0; innerRowOffsetA < BK; innerRowOffsetA += strideA) {
        if ((tileOffset + BK + innerRowOffsetA + innerRowA) < K &&
            (colOffsetA + innerColA * VECTOR_SIZE) < N) {
          __hip_fp8x4_e4m3_fnuz tmp = *(__hip_fp8x4_e4m3_fnuz *)&A[(tileOffset + BK + innerRowOffsetA + innerRowA) * N + 
                                         (colOffsetA + innerColA * VECTOR_SIZE)];
          *(__hip_fp8x4_e4m3_fnuz *)&As[next_buffer][innerRowOffsetA + innerRowA][innerColA * VECTOR_SIZE] = tmp;

          // Load A scale
          if (threadIdx.x < (BN / VECTOR_SIZE)) {
            float4 tmp = *(float4 *)&A_scale[((tileOffset + BK + innerRowOffsetA + innerRowA) / BLOCK_DIM) * N + 
                            (colOffsetA + innerColA * VECTOR_SIZE)];
            *(float4 *)&Ws[next_buffer][threadIdx.x * VECTOR_SIZE] = tmp;
          }
        } else {
          *(__hip_fp8x4_e4m3_fnuz *)&As[next_buffer][innerRowOffsetA + innerRowA][innerColA * VECTOR_SIZE] = float4{0.0f, 0.0f, 0.0f, 0.0f};
        }
      }

      // Load next B tile
      for (size_t innerRowOffsetB = 0; innerRowOffsetB < BK; innerRowOffsetB += strideB) {
        if ((tileOffset + BK + innerRowOffsetB + innerRowB) < K &&
            (colOffsetB + innerColB * VECTOR_SIZE) < M) {
          __hip_fp8x4_e4m3_fnuz tmp = *(__hip_fp8x4_e4m3_fnuz *)&B[(tileOffset + BK + innerRowOffsetB + innerRowB) * M + 
                                         (colOffsetB + innerColB * VECTOR_SIZE)];
          *(__hip_fp8x4_e4m3_fnuz *)&Bs[next_buffer][innerRowOffsetB + innerRowB][innerColB * VECTOR_SIZE] = tmp;

          // Load B scale
          if (threadIdx.x == 0) {
            Ws[next_buffer][BN] = B_scale[((tileOffset + BK + innerRowOffsetB + innerRowB) / BLOCK_DIM) * cdiv(M, BLOCK_DIM) + 
                                       ((colOffsetB + innerColB * VECTOR_SIZE) / BLOCK_DIM)];
          }
        } else {
          *(__hip_fp8x4_e4m3_fnuz *)&Bs[next_buffer][innerRowOffsetB + innerRowB][innerColB * VECTOR_SIZE] = float4{0.0f, 0.0f, 0.0f, 0.0f};
        }
      }
    }

    // Process current tile in chunks of 16 (MFMA input size)
    for (size_t BKOffset = 0; BKOffset < BK; BKOffset += 16) {
      // Load data into registers for MFMA
      for (size_t i = 0; i < 8; ++i) {
        a[i] = As[current_buffer][BKOffset + warpY * 8 + i][warpRowOffset + warpX];
        b[i] = Bs[current_buffer][BKOffset + warpY * 8 + i][warpColOffset + warpX];
      }

      // Execute MFMA operation
      floatx16 c = {0};
      c = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
          *reinterpret_cast<long *>(a), 
          *reinterpret_cast<long *>(b), 
          c, 0, 0, 0);

      // Apply scaling factors
      float b_scale = Ws[current_buffer][BN];
      for (size_t j = 0; j < 4; ++j) {
        for (size_t i = 0; i < 4; ++i) {
          d[i + j * 4] += c[i + j * 4] * Ws[current_buffer][warpRowOffset + j * 8 + warpY * 4 + i] * b_scale;
        }
      }
    }

    __syncthreads();
    
    // Swap buffers for next iteration
    current_buffer = 1 - current_buffer;
    next_buffer = 1 - next_buffer;
  }

  // Write results back to global memory
  for (size_t j = 0; j < 4; ++j) {
    for (size_t i = 0; i < 4; ++i) {
      if ((rowOffsetC + warpRowOffset + j * 8 + warpY * 4 + i) < N &&
          (colOffsetC + warpColOffset + warpX) < M) {
        C[(rowOffsetC + warpRowOffset + j * 8 + warpY * 4 + i) * M + 
          (colOffsetC + warpColOffset + warpX)] = (__hip_bfloat16)d[i + j * 4];
      }
    }
  }
}

// Flag to indicate if initialization has been done
bool initialized = false;

// Static stream for reuse
hipStream_t defaultStream = nullptr;

// Static events for timing and synchronization
hipEvent_t startEvent = nullptr;
hipEvent_t stopEvent = nullptr;

// Initialize resources and warm up the GPU
void initialize_resources() {
  if (!initialized) {
    // Create stream and events
    hipStreamCreate(&defaultStream);
    hipEventCreate(&startEvent);
    hipEventCreate(&stopEvent);
    
    // Warm up the GPU
    warmup_kernel<<<1, 256, 0, defaultStream>>>();
    hipStreamSynchronize(defaultStream);
    
    // Run a small dummy matrix multiply to initialize all code paths
    dim3 dummyThreads(256);
    dim3 dummyBlocks(1, 1);
    fp8_mm_kernel<BLOCK_N, BLOCK_K, BLOCK_M><<<dummyBlocks, dummyThreads, 0, defaultStream>>>(
        nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0);
    hipStreamSynchronize(defaultStream);
    
    initialized = true;
  }
}

at::Tensor fp8_mm(at::Tensor A, at::Tensor B, at::Tensor A_scale,
                  at::Tensor B_scale, at::Tensor C) {
  // Ensure resources are initialized
  if (!initialized) {
    initialize_resources();
  }
  
  size_t N = A.size(0), K = A.size(1), M = B.size(0);
  
  // Use the block dimensions from the fixed constants
  const size_t BK = BLOCK_K;
  const size_t BN = BLOCK_N;
  const size_t BM = BLOCK_M;
  
  // Define thread block size
  const size_t numThreads = (BN * BM) / 16;
  
  dim3 threadsPerBlock(numThreads);
  dim3 numBlocks(cdiv(M, BM), cdiv(N, BN));
  
  // Record start event
  hipEventRecord(startEvent, defaultStream);
  
  // Launch kernel
  fp8_mm_kernel<BN, BK, BM><<<numBlocks, threadsPerBlock, 0, defaultStream>>>(
      (__hip_fp8_e4m3_fnuz *)A.data_ptr(), (__hip_fp8_e4m3_fnuz *)B.data_ptr(),
      A_scale.data_ptr<float>(), B_scale.data_ptr<float>(),
      (__hip_bfloat16 *)C.data_ptr(), N, K, M);
  
  // Record stop event
  hipEventRecord(stopEvent, defaultStream);
  
  // Only synchronize on the stop event, not the entire stream
  hipEventSynchronize(stopEvent);
  
  return C;
}

// Clean up resources when no longer needed
void cleanup_resources() {
  if (initialized) {
    hipStreamSynchronize(defaultStream);
    
    if (startEvent != nullptr) {
      hipEventDestroy(startEvent);
      startEvent = nullptr;
    }
    
    if (stopEvent != nullptr) {
      hipEventDestroy(stopEvent);
      stopEvent = nullptr;
    }
    
    if (defaultStream != nullptr) {
      hipStreamDestroy(defaultStream);
      defaultStream = nullptr;
    }
    
    initialized = false;
  }
}
"""

cpp_src = r"""
at::Tensor fp8_mm(at::Tensor A, at::Tensor B, at::Tensor A_scale,
                  at::Tensor B_scale, at::Tensor C);
void cleanup_resources();
void initialize_resources();
"""

module = load_inline(
    name="fp8_mm",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["fp8_mm", "cleanup_resources", "initialize_resources"],
    verbose=True,
    extra_cuda_cflags=["-O3", "--offload-arch=gfx942", "-std=c++20", "-ffp-contract=fast"],
)

# Ensure the GPU and driver are initialized during import
module.initialize_resources()

def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    result = module.fp8_mm(a, b, a_scale, b_scale, c)
    return result

# Make sure to clean up resources when the module is unloaded
import atexit
atexit.register(module.cleanup_resources)

#Collected mean times (µs): [418.0, 99.1, 420.0, 51.0, 255.0, 468.0, 286.0, 394.0, 45.9, 946.0, 413.0, 451.0, 183.0, 1236.0, 2800.0, 1386.0, 420.0, 212.0]
#Previous Geometric mean (µs): 353.7947122881542
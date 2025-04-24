#!POPCORN leaderboard amd-fp8-mm
#!POPCORN gpus MI300

import os
from task import input_t, output_t
import sys 
if sys.stdout is None:
    sys.stdout = open('/dev/stdout', 'w')
if sys.stderr is None:
    sys.stderr = open('/dev/stderr', 'w')
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

// Block sizes
#define BLOCK_N 64
#define BLOCK_K 32
#define BLOCK_M 128

// Vector size for memory accesses
#define VECTOR_SIZE 4

template <const size_t BN, const size_t BK, const size_t BM>
__global__ void fp8_mm_kernel(const __hip_fp8_e4m3_fnuz *A,
                              const __hip_fp8_e4m3_fnuz *B,
                              const float *A_scale, const float *B_scale,
                              __hip_bfloat16 *C, size_t N, size_t K, size_t M) {
  using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

  static constexpr size_t WARPSIZE = 64;
  
  // Get row and column offsets for this thread block
  size_t rowOffsetC = blockIdx.y * BN;
  size_t colOffsetC = blockIdx.x * BM;
  
  // Calculate thread indices within the block
  size_t laneIdx = threadIdx.x % WARPSIZE;
  size_t warpIdx = threadIdx.x / WARPSIZE;
  
  // Calculate warp offsets within the block
  size_t warpColOffset = (warpIdx % (BM / 32)) * 32;
  size_t warpRowOffset = (warpIdx / (BM / 32)) * 32;
  size_t warpX = laneIdx % 32;
  size_t warpY = laneIdx / 32;

  // Shared memory for tiles
  __shared__ __hip_fp8_e4m3_fnuz As[BK][BN];
  __shared__ __hip_fp8_e4m3_fnuz Bs[BK][BM];
  __shared__ float A_scales[BN];
  __shared__ float B_scale_shared;

  // Registers for loading data
  __hip_fp8_e4m3_fnuz a[8], b[8];
  
  // Accumulator registers
  floatx16 d = {0};

  // Calculate thread's responsibility for loading
  size_t tIdx = threadIdx.x;
  
  // Loop over K dimension in blocks
  for (size_t k = 0; k < K; k += BK) {
    // Load A tile with coalesced accesses
    for (size_t i = tIdx; i < BN * BK; i += blockDim.x) {
      size_t row = i / BN;
      size_t col = i % BN;
      
      if ((k + row) < K && (rowOffsetC + col) < N) {
        As[row][col] = A[(k + row) * N + (rowOffsetC + col)];
      } else {
        As[row][col] = 0.0f;
      }
    }
    
    // Load B tile with coalesced accesses
    for (size_t i = tIdx; i < BK * BM; i += blockDim.x) {
      size_t row = i / BM;
      size_t col = i % BM;
      
      if ((k + row) < K && (colOffsetC + col) < M) {
        Bs[row][col] = B[(k + row) * M + (colOffsetC + col)];
      } else {
        Bs[row][col] = 0.0f;
      }
    }
    
    // Load A scales
    for (size_t i = tIdx; i < BN; i += blockDim.x) {
      if ((rowOffsetC + i) < N) {
        // Compute the proper scale index based on the block in K dimension
        size_t scale_k_idx = k / BLOCK_DIM;
        A_scales[i] = A_scale[scale_k_idx * N + (rowOffsetC + i)];
      } else {
        A_scales[i] = 0.0f;
      }
    }
    
    // Load B scale
    if (tIdx == 0) {
      size_t scale_k_idx = k / BLOCK_DIM;
      size_t scale_m_idx = colOffsetC / BLOCK_DIM;
      B_scale_shared = B_scale[scale_k_idx * cdiv(M, BLOCK_DIM) + scale_m_idx];
    }
    
    __syncthreads();
    
    // Process current tile in chunks of 16 (MFMA input size)
    for (size_t BKOffset = 0; BKOffset < BK; BKOffset += 16) {
      // Load data into registers for MFMA
      for (size_t i = 0; i < 8; ++i) {
        if (BKOffset + warpY * 8 + i < BK) {
          a[i] = As[BKOffset + warpY * 8 + i][warpRowOffset + warpX];
          b[i] = Bs[BKOffset + warpY * 8 + i][warpColOffset + warpX];
        } else {
          a[i] = 0.0f;
          b[i] = 0.0f;
        }
      }

      // Execute MFMA operation (keeping intrinsics intact)
      floatx16 c = {0};
      c = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
          *reinterpret_cast<long*>(&a), 
          *reinterpret_cast<long*>(&b), 
          c, 0, 0, 0);

      // Apply scaling factors (keeping scaling intact)
      float b_scale = B_scale_shared;
      for (size_t j = 0; j < 4; ++j) {
        for (size_t i = 0; i < 4; ++i) {
          size_t row_idx = warpRowOffset + j * 8 + warpY * 4 + i;
          if (row_idx < BN) {
            d[i + j * 4] += c[i + j * 4] * A_scales[row_idx] * b_scale;
          }
        }
      }
    }
    
    __syncthreads();
  }

  // Write results back to global memory with coalesced access pattern
  for (size_t j = 0; j < 4; ++j) {
    for (size_t i = 0; i < 4; ++i) {
      size_t row_idx = rowOffsetC + warpRowOffset + j * 8 + warpY * 4 + i;
      size_t col_idx = colOffsetC + warpColOffset + warpX;
      if (row_idx < N && col_idx < M) {
        C[row_idx * M + col_idx] = (__hip_bfloat16)d[i + j * 4];
      }
    }
  }
}

at::Tensor fp8_mm(at::Tensor A, at::Tensor B, at::Tensor A_scale,
                  at::Tensor B_scale, at::Tensor C) {
  size_t N = A.size(0), K = A.size(1), M = B.size(0);
  
  // Use the block dimensions from the fixed constants
  const size_t BK = BLOCK_K;
  const size_t BN = BLOCK_N;
  const size_t BM = BLOCK_M;
  
  // Define thread block size
  const size_t numThreads = 256; // Use standard thread count
  
  dim3 threadsPerBlock(numThreads);
  dim3 numBlocks(cdiv(M, BM), cdiv(N, BN));
  
  fp8_mm_kernel<BN, BK, BM><<<numBlocks, threadsPerBlock>>>(
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

#Collected mean times (µs): [493.0, 116.0, 530.0, 48.6, 231.0, 583.0, 253.0, 498.0, 55.9, 964.0, 409.0, 560.0, 196.0, 1211.0, 2830.0, 1366.0, 533.0, 191.0]
#Geometric mean (µs): 381.1460465149561
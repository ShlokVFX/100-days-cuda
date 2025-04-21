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
  __shared__ float Ws[BN+1];  // +1 for B_scale

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
        
        // Direct memory access for A
        for (size_t i = 0; i < VECTOR_SIZE; i++) {
          As[innerRowOffsetA + innerRowA][innerColA * VECTOR_SIZE + i] = 
              A[(tileOffset + innerRowOffsetA + innerRowA) * N + 
                (colOffsetA + innerColA * VECTOR_SIZE + i)];
        }
        
      } else if ((innerRowOffsetA + innerRowA) < BK &&
                 (innerColA * VECTOR_SIZE) < BN) {
        // Initialize to zero if out of bounds
        for (size_t i = 0; i < VECTOR_SIZE; i++) {
          As[innerRowOffsetA + innerRowA][innerColA * VECTOR_SIZE + i] = __hip_fp8_e4m3_fnuz(0.0f);
        }
      }
    }
    
    // Load B from global to shared memory
    for (size_t innerRowOffsetB = 0; innerRowOffsetB < BK;
         innerRowOffsetB += strideB) {
      if ((tileOffset + innerRowOffsetB + innerRowB) < K &&
          (colOffsetB + innerColB * VECTOR_SIZE) < M &&
          (innerRowOffsetB + innerRowB) < BK &&
          (innerColB * VECTOR_SIZE) < BM) {
        
        // Direct memory access for B
        for (size_t i = 0; i < VECTOR_SIZE; i++) {
          Bs[innerRowOffsetB + innerRowB][innerColB * VECTOR_SIZE + i] = 
              B[(tileOffset + innerRowOffsetB + innerRowB) * M + 
                (colOffsetB + innerColB * VECTOR_SIZE + i)];
        }
        
      } else if ((innerRowOffsetB + innerRowB) < BK &&
                 (innerColB * VECTOR_SIZE) < BM) {
        // Initialize to zero if out of bounds
        for (size_t i = 0; i < VECTOR_SIZE; i++) {
          Bs[innerRowOffsetB + innerRowB][innerColB * VECTOR_SIZE + i] = __hip_fp8_e4m3_fnuz(0.0f);
        }
      }
    }
    
    // Load A_scale
    if (threadIdx.x < (BN / VECTOR_SIZE)) {
      for (size_t i = 0; i < VECTOR_SIZE; i++) {
        size_t idx = ((tileOffset / BLOCK_DIM) * N) + (colOffsetA + threadIdx.x * VECTOR_SIZE + i);
        if (idx < N) {
          Ws[threadIdx.x * VECTOR_SIZE + i] = A_scale[idx];
        } else {
          Ws[threadIdx.x * VECTOR_SIZE + i] = 0.0f;
        }
      }
    }
    
    // Load B_scale
    if (threadIdx.x == 0) {
      size_t b_scale_idx = ((tileOffset / BLOCK_DIM) * cdiv(M, BLOCK_DIM)) + (colOffsetC / BLOCK_DIM);
      Ws[BN] = B_scale[b_scale_idx % (cdiv(K, BLOCK_DIM) * cdiv(M, BLOCK_DIM))];
    }
    
    __syncthreads();

    for (size_t BKOffset = 0; BKOffset < BK && (tileOffset + BKOffset < K); BKOffset += 16) {
      for (size_t i = 0; i < 8; ++i) {
        if ((BKOffset + warpY * 8 + i) < BK && (warpRowOffset + warpX) < BN) {
          a[i] = As[BKOffset + warpY * 8 + i][warpRowOffset + warpX];
        } else {
          a[i] = __hip_fp8_e4m3_fnuz(0.0f);
        }
        
        if ((BKOffset + warpY * 8 + i) < BK && (warpColOffset + warpX) < BM) {
          b[i] = Bs[BKOffset + warpY * 8 + i][warpColOffset + warpX];
        } else {
          b[i] = __hip_fp8_e4m3_fnuz(0.0f);
        }
      }
      
      floatx16 c = {0};
      c = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
          *reinterpret_cast<long *>(a), 
          *reinterpret_cast<long *>(b), 
          c, 0, 0, 0);
          
      float b_scale = Ws[BN];
      for (size_t j = 0; j < 4; ++j) {
        for (size_t i = 0; i < 4; ++i) {
          size_t ws_idx = warpRowOffset + j * 8 + warpY * 4 + i;
          if (ws_idx < BN) {
            d[i + j * 4] += c[i + j * 4] * Ws[ws_idx] * b_scale;
          }
        }
      }
    }

    __syncthreads();
  }

  // Write results to output matrix C
  for (size_t j = 0; j < 4; ++j) {
    for (size_t i = 0; i < 4; ++i) {
      size_t row = rowOffsetC + warpRowOffset + j * 8 + warpY * 4 + i;
      size_t col = colOffsetC + warpColOffset + warpX;
      if (row < N && col < M) {
        C[row * M + col] = (__hip_bfloat16)d[i + j * 4];
      }
    }
  }
}

at::Tensor fp8_mm(at::Tensor A, at::Tensor B, at::Tensor A_scale,
                  at::Tensor B_scale, at::Tensor C) {
  // Get dimensions from the input tensors
  size_t N = A.size(0);
  size_t K = A.size(1);
  size_t M = B.size(1);

  const size_t BK = 16;
  const size_t BN = 64;
  const size_t BM = 64;
  dim3 numThreads((BN * BM) / 16);
  dim3 numBlocks(cdiv(M, BM), cdiv(N, BN));
  
  hipDeviceSynchronize();
  fp8_mm_kernel<BN, BK, BM><<<numBlocks, numThreads>>>(
      (__hip_fp8_e4m3_fnuz *)A.data_ptr(), 
      (__hip_fp8_e4m3_fnuz *)B.data_ptr(),
      A_scale.data_ptr<float>(), 
      B_scale.data_ptr<float>(),
      (__hip_bfloat16 *)C.data_ptr(), 
      N, K, M);
  hipDeviceSynchronize();
  
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
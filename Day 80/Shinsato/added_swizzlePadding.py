#!POPCORN leaderboard amd-fp8-mm
#!POPCORN gpus MI300

import os
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "clang++"

cuda_src = R"""
#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/amd_detail/amd_hip_fp8.h>

__host__ __device__ __forceinline__ size_t cdiv(size_t a, size_t b) {
  return (a + b - 1) / b;
}

#define BLOCK_DIM 128
#define SWIZZLE_PAD 1  // 1-element padding per row to avoid bank conflicts

template <const size_t BN, const size_t BK, const size_t BM>
__global__ void fp8_mm_kernel(const __hip_fp8_e4m3_fnuz *A,
                              const __hip_fp8_e4m3_fnuz *B,
                              const float *A_scale, const float *B_scale,
                              __hip_bfloat16 *C, size_t N, size_t K, size_t M) {
  using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

  static constexpr size_t VECTOR_SIZE = 4;
  static constexpr size_t WARPSIZE = 64;
  static constexpr size_t numThreads = (BN * BM) / 16;
  static constexpr size_t strideA = numThreads / (BN / VECTOR_SIZE);
  static constexpr size_t strideB = numThreads / (BM / VECTOR_SIZE);

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

  // allocate with padding to avoid bank conflicts
  __shared__ __hip_fp8_e4m3_fnuz As[BK][BN + SWIZZLE_PAD];
  __shared__ __hip_fp8_e4m3_fnuz Bs[BK][BM + SWIZZLE_PAD];
  __shared__ float Ws[BN + 1];

  __hip_fp8_e4m3_fnuz a[8], b[8];
  floatx16 acc = {0};

  for (size_t tile = 0; tile < K; tile += BK) {
    // load A tile
    for (size_t r = 0; r < BK; r += strideA) {
      size_t row = r + innerRowA;
      size_t col = innerColA * VECTOR_SIZE;
      if (tile + row < K && colOffsetA + col < N) {
        __hip_fp8x4_e4m3_fnuz tmp = *(__hip_fp8x4_e4m3_fnuz *)&A[(tile + row) * N + (colOffsetA + col)];
        *(__hip_fp8x4_e4m3_fnuz *)&As[row][col] = tmp;
        if (threadIdx.x < (BN / VECTOR_SIZE)) {
          float4 sf = *(float4 *)&A_scale[((tile + row) / BLOCK_DIM) * N + (colOffsetA + col)];
          *(float4 *)&Ws[col] = sf;
        }
      } else {
        *(__hip_fp8x4_e4m3_fnuz *)&As[row][col] = float4{0,0,0,0};
      }
    }
    // load B tile
    for (size_t r = 0; r < BK; r += strideB) {
      size_t row = r + innerRowB;
      size_t col = innerColB * VECTOR_SIZE;
      if (tile + row < K && colOffsetB + col < M) {
        __hip_fp8x4_e4m3_fnuz tmp = *(__hip_fp8x4_e4m3_fnuz *)&B[(tile + row) * M + (colOffsetB + col)];
        *(__hip_fp8x4_e4m3_fnuz *)&Bs[row][col] = tmp;
        if (threadIdx.x == BN) {
          Ws[BN] = B_scale[((tile + row) / BLOCK_DIM) * cdiv(M, BLOCK_DIM) + ((colOffsetB + col) / BLOCK_DIM)];
        }
      } else {
        *(__hip_fp8x4_e4m3_fnuz *)&Bs[row][col] = float4{0,0,0,0};
      }
    }

    __syncthreads();

    // compute MFMA over sub-blocks
    for (size_t off = 0; off < BK; off += 16) {
#pragma unroll
      for (size_t i = 0; i < 8; ++i) {
        size_t ra = off + warpY * 8 + i;
        size_t ca = warpRowOffset + warpX;
        a[i] = As[ra][ca];
        b[i] = Bs[ra][warpColOffset + warpX];
      }
      floatx16 res = {0};
      res = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(*reinterpret_cast<long*>(a), *reinterpret_cast<long*>(b), res, 0,0,0);
      float bscale = Ws[BN];
#pragma unroll 2
      for (size_t j = 0; j < 4; ++j)
        for (size_t i = 0; i < 4; ++i)
          acc[i+j*4] += res[i+j*4] * Ws[warpRowOffset + j*8 + warpY*4 + i] * bscale;
    }

    __syncthreads();
  }

  // write back
#pragma unroll 2
  for (size_t j = 0; j < 4; ++j)
    for (size_t i = 0; i < 4; ++i) {
      size_t r = rowOffsetC + warpRowOffset + j*8 + warpY*4 + i;
      size_t c = colOffsetC + warpColOffset + warpX;
      if (r < N && c < M)
        C[r*M + c] = (__hip_bfloat16)acc[i+j*4];
    }
}

at::Tensor fp8_mm(at::Tensor A, at::Tensor B, at::Tensor A_scale, at::Tensor B_scale, at::Tensor C) {
  size_t N = A.size(0), K = A.size(1), M = B.size(0);
  const size_t BK = 16, BN = 64, BM = 64;
  dim3 threads((BN*BM)/16);
  dim3 blocks(cdiv(M, BM), cdiv(N, BN));
  fp8_mm_kernel<BN,BK,BM><<<blocks,threads>>>(
    reinterpret_cast<const __hip_fp8_e4m3_fnuz*>(A.data_ptr()),
    reinterpret_cast<const __hip_fp8_e4m3_fnuz*>(B.data_ptr()),
    A_scale.data_ptr<float>(), B_scale.data_ptr<float>(),
    reinterpret_cast<__hip_bfloat16*>(C.data_ptr()), N, K, M);
  return C;
}
"""

cpp_src = R"""
at::Tensor fp8_mm(at::Tensor A, at::Tensor B, at::Tensor A_scale, at::Tensor B_scale, at::Tensor C);
"""

module = load_inline(
    name="fp8_mm_swizzle_corrected",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["fp8_mm"],
    verbose=True,
    extra_cuda_cflags=["-O3", "--offload-arch=gfx942", "-std=c++20"],
)


def custom_kernel(data: input_t) -> output_t:
    a,b,a_scale,b_scale,c = data
    return module.fp8_mm(a,b,a_scale,b_scale,c)
 # Corrected swizzle indexing—should now match reference outputs while avoiding bank conflicts.
#Collected mean times (µs): [803.0, 229.0, 690.0, 83.0, 520.0, 1253.0, 585.0, 676.0, 107.0, 2340.0, 966.0, 1034.0, 380.0, 2870.0, 6490.0, 3230.0, 1030.0, 435.0]
#Geometric mean (µs): 758.7786341510256
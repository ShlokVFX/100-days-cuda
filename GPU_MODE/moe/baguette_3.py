#!POPCORN leaderboard amd-fp8-mm
from task import input_t, output_t
import torch
from torch.utils.cpp_extension import load_inline
import time
import os
import sys

if "PYTORCH_ROCM_ARCH" not in os.environ:
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx942:xnack-"

kernel_cpp = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <pybind11/pybind11.h>
#include <iostream>

#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/amd_detail/amd_hip_fp8.h>

typedef __hip_fp8_e4m3_fnuz fp8;
using vec4f = __attribute__((__vector_size__(4 * sizeof(float)))) float;

#define WARP_SIZE 64
#define N_WARP_Y_DEF 2
#define N_WARP_X_DEF 4
// Output Tile config
#define TILE_SIZE_Y_DEF 128
#define TILE_SIZE_X_DEF 128
#define TILE_K_DIM_DEF 128

constexpr int TILE_SIZE_Y = TILE_SIZE_Y_DEF;
constexpr int TILE_SIZE_X = TILE_SIZE_X_DEF;
constexpr int K_DIM = TILE_K_DIM_DEF;
constexpr int NBWARPS_Y = N_WARP_Y_DEF; // 4
constexpr int NBWARPS_X = N_WARP_X_DEF; // 2
// intrinsic config
constexpr int MATMUL_DIM_K = 32;
constexpr int MATMUL_DIM_NM = 16;
constexpr int BLOCK_DIM = NBWARPS_X * NBWARPS_Y * WARP_SIZE; // 512

using vec4u = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
using vec2u = __attribute__((__vector_size__(2 * sizeof(uint32_t)))) uint32_t;
using vec2short =
    __attribute__((__vector_size__(2 * sizeof(uint16_t)))) uint16_t;

__device__ void matmul16x16(const __hip_fp8_e4m3_fnuz *a,
                            const __hip_fp8_e4m3_fnuz *b, int k, int a_col,
                            int b_col, vec4f &c, int m, int n) {
  int tid = threadIdx.x % WARP_SIZE;
  int row = k + tid % 8 + (tid / 16) * 8;
  int col = ((tid % 16) / 8);
  int a_col_s = (a_col / 8 + col);
  int b_col_s = (b_col / 8 + col);
  int col_a_swizzled = ((a_col_s / 2) ^ (row % 8)) * 2 + a_col_s % 2;
  int col_b_swizzled = ((b_col_s / 2) ^ (row % 8)) * 2 + b_col_s % 2;

  auto packedA = reinterpret_cast<const uint64_t *>(a + row * m)[col_a_swizzled];
  auto packedB = reinterpret_cast<const uint64_t *>(b + row * m)[col_b_swizzled];

  c = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(packedA, packedB, c, 0, 0, 0);
}

template<bool boundCheck>
__device__ void loadLDS(int warpId, int laneId, int kId, fp8 *s_a, fp8 *s_b,
                        const fp8 *a, const fp8 *b, int M, int N) {
  
  const fp8 *ptr = a;
  fp8 *dst = s_a;
  int warpIdLocal = warpId%4;
  int stride = M;
  
  uint32_t col_offset = TILE_SIZE_X * blockIdx.x;
  if (warpId>=4){
    ptr=b;
    col_offset = TILE_SIZE_Y * blockIdx.y;
    dst=s_b;
    stride= N;
  }
  bool b_check = true;
  if constexpr (boundCheck){
    b_check = col_offset < stride;
  }
  
    constexpr int DIMY = TILE_SIZE_Y / 8;
    vec2u b_value[8] = {0};
    if (b_check){
    uint32_t row_offset = kId;
    for (int r = 0; r < 8; r++) {
      int col = laneId % DIMY;
      int row = r + 32 * warpIdLocal + 8 * (laneId / DIMY);
      
  
        b_value[r] = (reinterpret_cast<const vec2u *>(ptr + (row_offset + row) * stride + (col_offset + col * 8)))[0];
    }
  }
    vec2u result_b[8];
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j) {
        reinterpret_cast<uint8_t *>(&result_b[i])[j] =
            reinterpret_cast<uint8_t *>(&b_value[j])[i];
      }

    for (int r = 0; r < 8; r++) {

      int col = laneId % DIMY;
      int row = r + 32 * warpIdLocal + 8 * (laneId / DIMY);
      int col_swizzled = 2 * ((col / 2) ^ (row % 8)) + col % 2;

      (reinterpret_cast<vec2u *>(dst))[(DIMY)*row + col_swizzled] = result_b[r];
    }
  
}

__device__ float move_dpp(float v, int a, int b, int c, int d) {
  float result;
  asm volatile("v_mov_b32 %0, %1 quad_perm:[%2,%3,%4,%5]"
               : "=v"(result) // %0: output register
               : "v"(v), "i"(a), "i"(b), "i"(c), "i"(d));
  return result;
}
__global__ void cast_kernel(__hip_bfloat16 *c_bf16, const float *c_fp32, int M, int N) {

    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if (tid<M*N){
      c_bf16[tid] = (__hip_bfloat16)c_fp32[tid];
    }
}


template <int K_SPLIT,bool boundCheck, bool useAtomic>
__global__ void __launch_bounds__(N_WARP_X_DEF *N_WARP_Y_DEF *WARP_SIZE)
    custom_kernel(const fp8 *a, const fp8 *b, const float *as, const float *bs,
                  __hip_bfloat16 *c,float *c_atomic, int M, int N, int K) {
  int tid = threadIdx.x;
  int laneId = tid % WARP_SIZE;

  int splitK = K / K_SPLIT;
  int kIdStart = blockIdx.z * splitK;
  int kIdStop = splitK + blockIdx.z * splitK;

  int SN = (N + 128 - 1) / 128;

  int warpId = threadIdx.x / WARP_SIZE;
  int warpIdx = warpId / NBWARPS_Y; 
  int warpIdy = warpId % NBWARPS_Y;

  __shared__ fp8 s_a[K_DIM][TILE_SIZE_X];
  __shared__ fp8 s_b[K_DIM][TILE_SIZE_Y];

  constexpr int WARP_TILE_SIZE_Y = TILE_SIZE_Y / NBWARPS_Y / MATMUL_DIM_NM; // 2
  constexpr int WARP_TILE_SIZE_X = TILE_SIZE_X / NBWARPS_X / MATMUL_DIM_NM; // 4
  vec4f c_s[WARP_TILE_SIZE_X][WARP_TILE_SIZE_Y] = {0};

  for (int kId = kIdStart; kId < kIdStop; kId += K_DIM) {

    loadLDS<boundCheck>(warpId, laneId, kId, &s_a[0][0], &s_b[0][0], a, b, M, N);
    __syncthreads();

    vec4f c_tmp[WARP_TILE_SIZE_X][WARP_TILE_SIZE_Y] = {0};
#pragma unroll
    for (int t_wIdx = 0; t_wIdx < WARP_TILE_SIZE_X; t_wIdx++) {
      for (int t_wIdy = 0; t_wIdy < WARP_TILE_SIZE_Y; t_wIdy++) {
        constexpr int stepX = TILE_SIZE_X / WARP_TILE_SIZE_X;
        constexpr int stepY = TILE_SIZE_Y / WARP_TILE_SIZE_Y;

        vec4f c_tmp = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int k = 0; k < K_DIM;k += MATMUL_DIM_K) // MATMUL_DIM_K must be a multiple of K_DIM
        {
          int a_col = (t_wIdx * stepX) + warpIdx * MATMUL_DIM_NM;
          int b_col = (t_wIdy * stepY) + warpIdy * MATMUL_DIM_NM;
          matmul16x16(&s_a[0][0], &s_b[0][0], k, a_col, b_col, c_tmp,
                           TILE_SIZE_X, TILE_SIZE_Y);
        }

        int a_row = (t_wIdx * stepX) + warpIdx * MATMUL_DIM_NM;
        // Apply scaling
        int as_col = kId / 128;
        int as_row = a_row + blockIdx.x * TILE_SIZE_X;
        int bs_col = blockIdx.y;
        int bs_row = kId / 128;
        float bs_val = bs[bs_row * SN + bs_col];
        for (int i = 0; i < 4; i++) {
          int row = as_row + i + (laneId / 16) * 4;
          float as_val = as[as_col * M + row];
          c_s[t_wIdx][t_wIdy][i] += c_tmp[i] * as_val * bs_val;
        }
      }
    }

    __syncthreads();
  }
  

  for (int t_wIdx = 0; t_wIdx < WARP_TILE_SIZE_X; t_wIdx++) {
    for (int t_wIdy = 0; t_wIdy < WARP_TILE_SIZE_Y; t_wIdy++) {
      // offset pointing to the accumulated 16x16 block
      int col_offset =
          blockIdx.y * TILE_SIZE_Y +                 // TILE offset
          (warpIdy * MATMUL_DIM_NM) +                // Warp offset
          t_wIdy * (TILE_SIZE_Y / WARP_TILE_SIZE_Y); // Warp tile offset

      int row_offset =
          blockIdx.x * TILE_SIZE_X +                 // TILE offset
          (warpIdx * MATMUL_DIM_NM) +                // Warp offset
          t_wIdx * (TILE_SIZE_X / WARP_TILE_SIZE_X); // Warp tile offset

      auto res = c_s[t_wIdx][t_wIdy];

      bool b_check = true;
      if constexpr (boundCheck){
        b_check = (row_offset) < M && (col_offset) < N;
      }
      if (b_check) {
          // C is stored on 4 VGPRs
          for (int i = 0; i < 4; i++)
          {
              int col = laneId % 16;
              int row = i + (laneId / 16) * 4;
              if constexpr (useAtomic){
                atomicAdd( &c_atomic[(row_offset + row) * N + (col_offset + col)],res[i]);
              }else{
                c[(row_offset + row) * N + (col_offset + col)] = (__hip_bfloat16)(res[i]);
              }
          }
        
      }
    }
  }
}


struct LaunchConfig {
  int K_SPLIT;
  bool useAtomic;
  bool boundCheck;
};

template <int K_SPLIT, bool boundCheck, bool useAtomic>
inline void launch_kernel_variant(const dim3& blocks, const dim3& threads,
                                  const fp8 *d_A, const fp8 *d_B,
                                  const float *d_As, const float *d_Bs, __hip_bfloat16 *d_C,float*c_atomic,
                                  int M, int N, int K) {

  dim3 threadsCast(256);
  dim3 blockCast((M*N + (N*M-1))/256);

  hipLaunchKernelGGL((custom_kernel<K_SPLIT, boundCheck, useAtomic>), blocks, threads, 0, 0,
                     d_A, d_B, d_As, d_Bs, d_C, c_atomic,M, N, K);
  if constexpr (useAtomic){
    hipLaunchKernelGGL(cast_kernel, blockCast,threadsCast,0,0,d_C,c_atomic,M,N);
  }
}


template <int K_SPLIT>
inline void dispatch_bound_atomic(bool boundCheck, bool useAtomic,
                                  const dim3& blocks, const dim3& threads,
                                  const fp8 *d_A, const fp8 *d_B,
                                  const float *d_As, const float *d_Bs, __hip_bfloat16 *d_C,float*c_atomic,
                                  int M, int N, int K) {
  if (boundCheck) {
    if (useAtomic)
    {
      hipMemset(c_atomic, 0, N*M * sizeof(float));
      launch_kernel_variant<K_SPLIT, true, true>(blocks, threads, d_A, d_B, d_As, d_Bs, d_C,c_atomic, M, N, K);
    }
    else
      launch_kernel_variant<K_SPLIT, true, false>(blocks, threads, d_A, d_B, d_As, d_Bs, d_C,c_atomic, M, N, K);
  } else {
    if (useAtomic){
      hipMemset(c_atomic, 0, N*M * sizeof(float));
      launch_kernel_variant<K_SPLIT, false, true>(blocks, threads, d_A, d_B, d_As, d_Bs, d_C,c_atomic, M, N, K);
    }
    else
      launch_kernel_variant<K_SPLIT, false, false>(blocks, threads, d_A, d_B, d_As, d_Bs, d_C,c_atomic, M, N, K);
  }
}

void launch_kernel(const LaunchConfig &config, const fp8 *d_A, const fp8 *d_B,
                   const float *d_As, const float *d_Bs, __hip_bfloat16 *d_C,float*c_atomic,
                   int M, int N, int K) {

  dim3 threadsPerBlock(N_WARP_X_DEF * N_WARP_Y_DEF * WARP_SIZE);
  dim3 blocksPerGrid((M + TILE_SIZE_X_DEF - 1) / TILE_SIZE_X_DEF,
                     (N + TILE_SIZE_Y_DEF - 1) / TILE_SIZE_Y_DEF,
                     config.K_SPLIT);

  switch (config.K_SPLIT) {
    case 1: dispatch_bound_atomic<1>(config.boundCheck, config.useAtomic, blocksPerGrid, threadsPerBlock, d_A, d_B, d_As, d_Bs, d_C,c_atomic, M, N, K); break;
    case 2: dispatch_bound_atomic<2>(config.boundCheck, config.useAtomic, blocksPerGrid, threadsPerBlock, d_A, d_B, d_As, d_Bs, d_C,c_atomic, M, N, K); break;
    case 4: dispatch_bound_atomic<4>(config.boundCheck, config.useAtomic, blocksPerGrid, threadsPerBlock, d_A, d_B, d_As, d_Bs, d_C,c_atomic, M, N, K); break;
    default:
      throw std::invalid_argument("Unsupported K_SPLIT value");
  }
}


void launch_kernel(const fp8 *d_A, const fp8 *d_B, const float *d_As,
                   const float *d_Bs, __hip_bfloat16 *d_C, float*c_atomic,int M, int N,
                   int K) {
  LaunchConfig config;
  config.useAtomic=false;
  config.K_SPLIT=1;
  if (M % 128 ==0 && N %128 ==0)
    config.boundCheck = false;
  else
    config.boundCheck = true;

  //SPLIT K for these :
  if ((M == 1024 &&  N==1536 && K ==7168 )||
      (M == 1024 &&  N==512 && K ==7168 ) ||
      (M == 1024 &&  N==576 && K ==7168 )){
        config.useAtomic = true;
        config.K_SPLIT = 4;
      }  
  launch_kernel(config, d_A, d_B, d_As, d_Bs, d_C,c_atomic, M, N, K);
}



#define HIP_CHECK(condition)                                                                  @
    do                                                                                        @
    {                                                                                         @
        hipError_t error = condition;                                                         @
        if(error != hipSuccess)                                                               @
        {                                                                                     @
            std::cout << "Error " << hipGetErrorName(error) << '(' << error << ')' << ": "    @
                      << hipGetErrorString(error) << " in " << __func__ << " at " << __FILE__ @
                      << ':' << __LINE__ << '@n';                                             @
            exit(error);                                                                      @
        }                                                                                     @
    }                                                                                         @
    while(false)

void run(
    uintptr_t a_ptr,
    uintptr_t b_ptr,
    uintptr_t as_ptr,
    uintptr_t bs_ptr,
    uintptr_t c_ptr,
    int M,
    int N,
    int K
) {
    const auto* d_A = reinterpret_cast<const fp8*>(a_ptr);
    const auto* d_B = reinterpret_cast<const fp8*>(b_ptr);
    const auto* d_As = reinterpret_cast<const float*>(as_ptr);
    const auto* d_Bs = reinterpret_cast<const float*>(bs_ptr);
    auto* d_C = reinterpret_cast<__hip_bfloat16*>(c_ptr);

    dim3 threadsPerBlock(N_WARP_X_DEF * N_WARP_Y_DEF * WARP_SIZE);
    dim3 blocksPerGrid((M + TILE_SIZE_X_DEF - 1) / TILE_SIZE_X_DEF,
                     (N + TILE_SIZE_Y_DEF - 1) / TILE_SIZE_Y_DEF);
    
    static float* d_array = nullptr;
    static int n=-1;
    static int m=-1;
    if (!d_array)
    {
        size_t size = 6144*7168 * sizeof(float);
        hipError_t err = hipMalloc(&d_array, size);
    }
   // hipMemset(d_array, 0, N*M * sizeof(float));
    launch_kernel(d_A, d_B, d_As,
                   d_Bs,d_C, d_array, M, N, K); 

    //hipFree(d_array);
    HIP_CHECK(hipGetLastError());
}

PYBIND11_MODULE(fp8, m) {
  m.def("fp8", &run, "HIP kernel");
}
"""

hip_module = load_inline(
    name="fp8",
    cpp_sources="",
    cuda_sources=kernel_cpp.replace('@', chr(92)),
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["-g --save-temps -Rpass-analysis=kernel-resource-usage -std=c++20"],
    no_implicit_headers=True,
)

first = True

def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of block-scale fp8 gemm
    Args:
        data: Tuple that expands to:
            a: torch.Tensor[float8_e4m3fnuz] of shape [m, k],
            b: torch.Tensor[float8_e4m3fnuz] of shape [n, k],
            a_scale: torch.Tensor[float32] of shape [m, k // 128],
            b_scale: torch.Tensor[float32] of shape [n // 128, k // 128],
            c: torch.Tensor[bfloat16] of shape [m, n]
    Returns:
        Tensor containing output in bf16
    """

    global first
    if first:
        print("executing on", torch.cuda.get_device_name(), file=sys.stderr)
        first = False

    a, b, a_scale, b_scale, c = data

    m, n = c.shape
    k = a.shape[1]

    hip_module.fp8(
        a.data_ptr(),
        b.data_ptr(),
        a_scale.data_ptr(),
        b_scale.data_ptr(),
        c.data_ptr(),
        m,
        n,
        k
    )

    return c

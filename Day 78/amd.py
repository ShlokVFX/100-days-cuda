#!POPCORN leaderboard amd-fp8-mm

from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
CPP_WRAPPER = """
void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c);
"""

CUDA_SRC = """
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

// Optimized constants for AMD MI300X
constexpr const int BM = 64;      // Block size for M dimension
constexpr const int BN = 64;      // Block size for N dimension
constexpr const int BK = 128;     // Block size for K dimension (matching scaling granularity)
constexpr const int THREADS = 256; // Threads per block

__global__ void optimized_fp8_gemm_kernel(
    const __hip_fp8_e4m3_fnuz* __restrict__ a,      // M x K (column-major)
    const __hip_fp8_e4m3_fnuz* __restrict__ b,      // N x K (column-major)
    const float* __restrict__ a_scale,              // M x (K/128) scaling factors
    const float* __restrict__ b_scale,              // N x (K/128) scaling factors
    __hip_bfloat16* __restrict__ c,                 // M x N (row-major) output
    int m, int n, int k) {
    
    // Thread position
    int tid = threadIdx.x;
    
    // Block position in grid
    int bm = blockIdx.x;
    int bn = blockIdx.y;
    
    // Global block positions
    int m_start = bm * BM;
    int n_start = bn * BN;
    
    // Thread position within block (each thread handles 4x4 output elements)
    int tm = tid % 16;
    int tn = tid / 16;
    
    // Thread's global positions
    int m_pos = m_start + tm * 4;
    int n_pos = n_start + tn * 4;
    
    // Create local accumulator for 4x4 block
    float local_c[4][4] = {0.0f};
    
    // Process input matrices in blocks of K dimension
    for (int kb = 0; kb < k; kb += BK) {
        // Shared memory for input tiles and scaling factors
        __shared__ float a_tile[BM][BK];
        __shared__ float b_tile[BK][BN];
        __shared__ float a_scale_shared[BM];
        __shared__ float b_scale_shared[BN];
        
        // Current K block index for scaling (each scale applies to 128 elements)
        int scale_block = kb / 128;
        
        // Collaboratively load a_scale for this block
        for (int i = tid; i < BM; i += THREADS) {
            int global_m = m_start + i;
            if (global_m < m) {
                // Column-major access pattern for scales: row + col*m
                a_scale_shared[i] = a_scale[global_m + scale_block * m];
            } else {
                a_scale_shared[i] = 1.0f; // Default for out-of-bounds
            }
        }
        
        // Collaboratively load b_scale for this block
        for (int i = tid; i < BN; i += THREADS) {
            int global_n = n_start + i;
            if (global_n < n) {
                // Column-major access pattern for scales: row + col*n
                b_scale_shared[i] = b_scale[global_n + scale_block * n];
            } else {
                b_scale_shared[i] = 1.0f; // Default for out-of-bounds
            }
        }
        
        // Load a_tile collaboratively (convert FP8 to float)
        for (int i = tid; i < BM * BK; i += THREADS) {
            int local_m = i % BM;
            int local_k = i / BM;
            
            int global_m = m_start + local_m;
            int global_k = kb + local_k;
            
            if (global_m < m && global_k < k) {
                // Column-major: a[row + col*m]
                a_tile[local_m][local_k] = (float)a[global_m + global_k * m];
            } else {
                a_tile[local_m][local_k] = 0.0f;
            }
        }
        
        // Load b_tile collaboratively (convert FP8 to float)
        for (int i = tid; i < BK * BN; i += THREADS) {
            int local_k = i % BK;
            int local_n = i / BK;
            
            int global_k = kb + local_k;
            int global_n = n_start + local_n;
            
            if (global_k < k && global_n < n) {
                // Column-major: b[row + col*n]
                b_tile[local_k][local_n] = (float)b[global_n + global_k * n];
            } else {
                b_tile[local_k][local_n] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute matrix multiplication for this K block
        if (tm < 16 && tn < 16) { // Only threads within bounds compute output
            // Perform 4x4 block matrix multiplication
            for (int k_idx = 0; k_idx < BK && kb + k_idx < k; k_idx++) {
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        // Accumulate in higher precision
                        local_c[i][j] += a_tile[tm * 4 + i][k_idx] * b_tile[k_idx][tn * 4 + j];
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Apply scaling factors to final result
    if (tm < 16 && tn < 16) {
        // Calculate number of K blocks for scaling
        int num_k_blocks = (k + 127) / 128;
        
        // For each position in 4x4 block, apply scaling and write to output
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                int global_m = m_pos + i;
                int global_n = n_pos + j;
                
                if (global_m < m && global_n < n) {
                    // Apply all applicable scale factors
                    float scaled_result = 0.0f;
                    
                    // Process each K block's contribution with proper scaling
                    for (int kb = 0; kb < num_k_blocks; kb++) {
                        float a_scale_val = a_scale[global_m + kb * m];
                        float b_scale_val = b_scale[global_n + kb * n];
                        
                        // Apply scaling for this K block using the product of scales
                        float scale_factor = a_scale_val * b_scale_val;
                        scaled_result += local_c[i][j] * scale_factor;
                    }
                    
                    // Write to output in row-major format: c[row*n + col]
                    c[global_m * n + global_n] = (__hip_bfloat16)scaled_result;
                }
            }
        }
    }
}

void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor a_scale, torch::Tensor b_scale, torch::Tensor c) {
    int m = a.size(0);
    int n = b.size(0);
    int k = a.size(1);
    
    // Calculate grid dimensions for kernel launch
    dim3 grid((m + BM - 1) / BM, (n + BN - 1) / BN, 1);
    dim3 block(THREADS, 1, 1);
    
    // Launch kernel with explicit pointer casting to prevent type errors
    hipLaunchKernelGGL(optimized_fp8_gemm_kernel, grid, block, 0, 0,
                      (__hip_fp8_e4m3_fnuz*)a.data_ptr(), 
                      (__hip_fp8_e4m3_fnuz*)b.data_ptr(),
                      (const float*)a_scale.data_ptr(),  // Cast needed to fix compilation error
                      (const float*)b_scale.data_ptr(),  // Cast needed to fix compilation error
                      (__hip_bfloat16*)c.data_ptr(), 
                      m, n, k);
}

"""

import os
os.environ["CXX"] = "clang++"

module = load_inline(
    name='fp8_mm',
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    functions=['fp8_mm'],
    verbose=True,
    extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20"],
)


def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    module.fp8_mm(a, b, a_scale, b_scale, c)
    return c



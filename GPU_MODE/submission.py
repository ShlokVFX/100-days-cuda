#!POPCORN leaderboard amd-fp8-mm

# This is a submission template for popcorn leaderboard 'amd-fp8-mm'.
# Your task is as follows:
# > 
# > You will implement a custom fp8-blockwise matmul kernel optimized for MI300.
# > You will be given single-precision scaling factors for your matrices.
# > The shapes of all outer and inner dimensions of tensors are from DeepSeek-R1.
# > To be explicit, you will be given a tuple of tensors:
# > ```
# > (a, b, a_scale, b_scale, c)
# > ```
# > where `a` and `b` are the input matrices, `a_scale` and `b_scale` are the scaling factors for `a` and `b` respectively,
# > and `c` is the output matrix:
# > * `a` is M x K in column-major order in e4m3fnuz
# > * `b` is N x K in column-major order in e4m3fnuz
# > * `a_scale` is M x K // 128 in column-major order in fp32
# > * `b_scale` is N // 128 x K // 128 in column-major order in fp32
# > * `c` is M x N in ROW-major order in bf16
# > 
# > Matrix sizes `m` and `n` are divisible by 64, `k` is divisible by 128.
# > 
# > The ranking criteria is the geometric mean of the benchmark results.
# > 
# > For the grand price, your kernel will be evaluated against the speed of light analysis
# > and the solution closest to the speed of light will be awarded the grand price.
# > ```
# > The speed of light analysis is:
# >  M       N       K     time[us]
# > 1024    1536    7168      8.63
# > 1024    4608    7168     25.89
# > 6144    1536    7168     51.78
# > 6144    4608    7168    155.30
# > 1024    7168     256      3.17
# > 6144    7168     256     17.27
# > ```
# The deadline for this leaderboard is 2025-05-27 00:00:00+00:00

# You can automatically route this file to specific GPUs by adding a line
# `#!POPCORN gpus <GPUs>` to the header of this file.
# Happy hacking!

# This script provides a template for using load_inline to run a HIP kernel for
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
CPP_WRAPPER = """
void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c);
"""

CUDA_SRC = """
#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/amd_detail/amd_hip_bf16.h>

constexpr const int BLOCK = 128;

__global__ void custom_kernel(const __hip_fp8_e4m3_fnuz* a, const __hip_fp8_e4m3_fnuz* b, const float* as, const float* bs, 
                   __hip_bfloat16* c, int m, int n, int k) {
                   
    // Your implementation here
    int cx = threadIdx.x + blockDim.x * blockIdx.x;
    int cy = threadIdx.y + blockDim.y * blockIdx.y;
    if(cx >= m || cy >= n) return;
    
    int sn = (n + BLOCK - 1) / BLOCK;
    
    float result = 0;
    // split loop into an outer loop over different blocks, and an inner loop within one block.
    // we can assume k % BLOCK == 0.
    for(int i = 0; i < k; i += BLOCK) {
        // block results accumulates the inner product across a single block.
        // within each block, scales are constant, so we can lift the scaling 
        // outside of the inner loop.
        float block_result = 0;
        for(int ii = 0; ii < BLOCK; ++ii) {
            // load input matrix elements and convert to float for computations
            float av = (float)a[cx + (i + ii) * m];
            float bv = (float)b[cy + (i + ii) * n];
            block_result += av * bv; 
        }
        
        // before we can go to the next block, scale the result of the current block
        // and accumulate to final result
        // note the different indexing into as and bs
        result += block_result * as[cx + i/BLOCK * m] * bs[cy/BLOCK + i/BLOCK * sn];
    }
    
    // finally, write the result as bf16
    c[cx * n + cy] = (__hip_bfloat16)result;
}

void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c) {
    int m = a.size(0);
    int n = b.size(0);
    int k = a.size(1);
    custom_kernel<<<dim3((m+15)/16, (n+15)/16), dim3(16, 16), 0, 0>>> ((__hip_fp8_e4m3_fnuz*)a.data_ptr(), (__hip_fp8_e4m3_fnuz*)b.data_ptr(), 
    as.data_ptr<float>(), bs.data_ptr<float>(), (__hip_bfloat16*)c.data_ptr(), m, n, k);
    //C10_CUDA_CHECK(cudaGetLastError());
}
"""

import os
os.environ["CXX"] = "clang++"
import sys
if sys.stdout is None:
    sys.stdout = open("/dev/stdout", "w")
if sys.stderr is None:
    sys.stderr = open("/dev/stderr", "w")

module = load_inline(
    name='fp8_mm',
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    functions=['fp8_mm'],
    verbose=True,
   # extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20"],
)


def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    module.fp8_mm(a, b, a_scale, b_scale, c)
    return c



#include <cuda_runtime.h>
#include <iostream>

#define N 1048576  // Large enough to maximize occupancy
#define THREADS_PER_BLOCK 256  // Optimal block size

__global__ void optimized_reduction(float* input, float* output) {
    __shared__ float shared_data[THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory (avoid extra reads)
    float local_sum = (global_id < N) ? input[global_id] : 0.0f;
    shared_data[tid] = local_sum;
    __syncthreads();

    // Reduction within a block using shared memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Final warp-level reduction (no need for __syncthreads() here)
    if (tid < 32) {
        float val = shared_data[tid];
        val += __shfl_down_sync(0xFFFFFFFF, val, 16);
        val += __shfl_down_sync(0xFFFFFFFF, val, 8);
        val += __shfl_down_sync(0xFFFFFFFF, val, 4);
        val += __shfl_down_sync(0xFFFFFFFF, val, 2);
        val += __shfl_down_sync(0xFFFFFFFF, val, 1);

        if (tid == 0) output[blockIdx.x] = val;
    }
}

void reduce(float* d_input, float* d_output, int num_elements) {
    int blocks = (num_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    optimized_reduction<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output);
}

int main() {
    float *d_input, *d_output;
    size_t size = N * sizeof(float);
    size_t output_size = ((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) * sizeof(float);
    
    // Allocate memory
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, output_size);
    
    // Launch optimized reduction kernel
    reduce(d_input, d_output, N);
    
    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}

#include <cuda_runtime.h>
#include <iostream>

#define N 1048576  // Large enough to maximize occupancy
#define THREADS_PER_BLOCK 256  // Block size

__global__ void naive_reduction(float* input, float* output) {
    __shared__ float shared_data[THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    shared_data[tid] = (global_id < N) ? input[global_id] : 0.0f;
    __syncthreads();

    // Naïve reduction with warp divergence
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {  // 🚨 Causes warp divergence
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();  // 🚨 Unnecessary synchronization for inactive threads
    }

    // Store block result
    if (tid == 0) output[blockIdx.x] = shared_data[0];
}

void reduce(float* d_input, float* d_output, int num_elements) {
    int blocks = (num_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    naive_reduction<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output);
}

int main() {
    float *d_input, *d_output;
    size_t size = N * sizeof(float);
    size_t output_size = ((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) * sizeof(float);
    
    // Allocate memory
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, output_size);
    
    // Launch naive reduction kernel
    reduce(d_input, d_output, N);
    
    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}

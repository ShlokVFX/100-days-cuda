#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256  // Optimize based on GPU
#define WARP_SIZE 32    // Standard warp size

// Warp reduction for max operation
__device__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp reduction for sum operation
__device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized Softmax Kernel
__global__ void softmaxOptimized(float* input, float* output, int rows, int cols) {
    extern __shared__ float shared_data[];

    int row = blockIdx.x;  // Each block processes one row
    int tid = threadIdx.x;

    if (row >= rows) return;

    float max_val = -INFINITY;
    for (int i = tid; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, input[row * cols + i]);
    }

    // Block-wide reduction for max
    __shared__ float block_max;
    if (tid == 0) block_max = -INFINITY;
    __syncthreads();
    
    atomicMax((int*)&block_max, __float_as_int(max_val));
    __syncthreads();
    
    max_val = block_max;

    // Compute exponentials and sum
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        shared_data[i] = expf(input[row * cols + i] - max_val);
        sum += shared_data[i];
    }

    // Block-wide reduction for sum
    __shared__ float block_sum;
    if (tid == 0) block_sum = 0.0f;
    __syncthreads();
    
    atomicAdd(&block_sum, sum);
    __syncthreads();
    
    sum = block_sum;

    // Normalize values
    for (int i = tid; i < cols; i += blockDim.x) {
        output[row * cols + i] = shared_data[i] / sum;
    }
}

// Host function to launch kernel
void softmax(float* h_input, float* h_output, int rows, int cols) {
    float *d_input, *d_output;
    size_t size = rows * cols * sizeof(float);

    // Allocate device memory
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Copy data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Launch optimized kernel
    int shared_mem_size = cols * sizeof(float);
    softmaxOptimized<<<rows, BLOCK_SIZE, shared_mem_size>>>(d_input, d_output, rows, cols);
    
    // Copy result back
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

// Test function
int main() {
    const int rows = 2, cols = 4;
    float h_input[rows * cols] = {1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0};
    float h_output[rows * cols];

    // Compute softmax
    softmax(h_input, h_output, rows, cols);

    // Print result
    std::cout << "Softmax Output:\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << h_output[i * cols + j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}

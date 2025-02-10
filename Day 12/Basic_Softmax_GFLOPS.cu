#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256  // Adjust based on your GPU capabilities

// Softmax kernel (row-wise computation)
__global__ void softmaxKernel(float* input, float* output, int rows, int cols) {
    extern __shared__ float shared_data[];
    
    int row = blockIdx.x;  // Each block handles one row
    int tid = threadIdx.x;
    
    if (row >= rows) return;

    float* row_data = input + row * cols;  // Pointer to current row
    float* shared_row = shared_data;  // Shared memory pointer

    // Step 1: Find max element in the row (for numerical stability)
    float max_val = -INFINITY;
    for (int i = tid; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, row_data[i]);
    }

    // Reduce max across threads in the block
    __shared__ float block_max;
    if (tid == 0) block_max = -INFINITY;
    __syncthreads();

    atomicMax((int*)&block_max, __float_as_int(max_val));
    __syncthreads();

    // Step 2: Compute exponentials and sum them
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        shared_row[i] = expf(row_data[i] - block_max);
        sum += shared_row[i];
    }

    // Reduce sum across threads
    __shared__ float block_sum;
    if (tid == 0) block_sum = 0.0f;
    __syncthreads();

    atomicAdd(&block_sum, sum);
    __syncthreads();

    // Step 3: Normalize the values
    for (int i = tid; i < cols; i += blockDim.x) {
        output[row * cols + i] = shared_row[i] / block_sum;
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

        // Create events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    
        // Record the start time
        cudaEventRecord(start);

    // Launch kernel (1 block per row)
    softmaxKernel<<<rows, BLOCK_SIZE, cols * sizeof(float)>>>(d_input, d_output, rows, cols);

     // Record the stop time
     cudaEventRecord(stop);
     cudaEventSynchronize(stop);
 
     // Calculate execution time in milliseconds
     float elapsedTime;
     cudaEventElapsedTime(&elapsedTime, start, stop);
     float timeInSeconds = elapsedTime / 1000.0f;
 
    
    // Copy result back
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

       // Calculate GFLOPS
       long long numFLOPs = (long long)rows * cols * 2;  // 2 FLOPs per element: exp and division
       float gflops = (numFLOPs / timeInSeconds) / 1e9;
   
       // Output the results
       std::cout << "Execution time: " << elapsedTime << " ms\n";
       std::cout << "GFLOPS: " << gflops << " GFLOPS\n";
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

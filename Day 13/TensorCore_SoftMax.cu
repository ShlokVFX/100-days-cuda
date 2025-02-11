#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // FP16 support for Tensor Cores

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Warp reduction for max operation (FP16)
__device__ __half warpReduceMax(__half val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = __hmax(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp reduction for sum operation (FP16)
__device__ __half warpReduceSum(__half val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = __hadd(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Optimized Softmax Kernel using Tensor Cores (FP16)
__global__ void softmaxTensorCore(__half* input, __half* output, int rows, int cols) {
    extern __shared__ __half shared_data[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    __half* row_data = input + row * cols;
    __half* shared_row = shared_data;

    // Step 1: Compute max for numerical stability
    __half max_val = __float2half(-INFINITY);
    for (int i = tid; i < cols; i += blockDim.x) {
        max_val = __hmax(max_val, row_data[i]);
    }
    max_val = warpReduceMax(max_val);

    // Step 2: Compute exponentials and sum
    __half sum = __float2half(0.0f);
    for (int i = tid; i < cols; i += blockDim.x) {
        shared_row[i] = __float2half(expf(__half2float(__hsub(row_data[i], max_val))));  // ✅ Corrected
        sum = __hadd(sum, shared_row[i]);
    }
    sum = warpReduceSum(sum);

    // Step 3: Normalize values
    for (int i = tid; i < cols; i += blockDim.x) {
        output[row * cols + i] = __hdiv(shared_row[i], sum);
    }
}

// Host function to launch kernel
void softmaxFP16(float* h_input, float* h_output, int rows, int cols) {
    __half *d_input, *d_output;
    size_t size = rows * cols * sizeof(__half);

    // Allocate device memory
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Convert FP32 input to FP16
    __half* h_input_half = new __half[rows * cols];
    for (int i = 0; i < rows * cols; i++)
        h_input_half[i] = __float2half(h_input[i]);

    // Copy data to device
    cudaMemcpy(d_input, h_input_half, size, cudaMemcpyHostToDevice);

      // Create events for timing
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
  
      // Record the start time
      cudaEventRecord(start);

    // Launch optimized softmax kernel
    int shared_mem_size = cols * sizeof(__half);
    softmaxTensorCore<<<rows, BLOCK_SIZE, shared_mem_size>>>(d_input, d_output, rows, cols);

     // Record the stop time
     cudaEventRecord(stop);
     cudaEventSynchronize(stop);

      // Calculate execution time in milliseconds
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    float timeInSeconds = elapsedTime / 1000.0f;


    // Copy result back to host
    __half* h_output_half = new __half[rows * cols];
    cudaMemcpy(h_output_half, d_output, size, cudaMemcpyDeviceToHost);

    // Convert FP16 output to FP32
    for (int i = 0; i < rows * cols; i++)
        h_output[i] = __half2float(h_output_half[i]);

    // Cleanup
    delete[] h_input_half;
    delete[] h_output_half;
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

    // Compute softmax using Tensor Core acceleration
    softmaxFP16(h_input, h_output, rows, cols);

    // Print result
    std::cout << "Softmax Output (FP16 Accelerated):\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << h_output[i * cols + j] << " ";
        }
        std::cout << "\n";
    }

    return 0;

}

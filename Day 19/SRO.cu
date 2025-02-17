#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

#define N (1 << 26)        // 67 million elements
#define THREADS_PER_BLOCK 256
#define ITERATIONS 100     // Run multiple iterations for averaging

__global__ void sumReductionOptimized(float *input, float *output, int n) {
    __shared__ float shared[THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float sum = 0.0f;
    
    if (idx < n)
        sum += input[idx];
    if (idx + blockDim.x < n)
        sum += input[idx + blockDim.x];

    shared[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    
    if (tid == 0)
        output[blockIdx.x] = shared[0];
}

float finalReductionOnCPU(float *d_output, int size) {
    float *h_partial_sums = new float[size];
    cudaMemcpy(h_partial_sums, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    float total_sum = 0.0f;
    for (int i = 0; i < size; i++) {
        total_sum += h_partial_sums[i];
    }
    delete[] h_partial_sums;
    return total_sum;
}

int main() {
    // Allocate and initialize host memory
    float *h_input = new float[N];
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    int blocks = (N + THREADS_PER_BLOCK * 2 - 1) / (THREADS_PER_BLOCK * 2);
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, blocks * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Warm-up for the custom kernel
    sumReductionOptimized<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    // Create CUDA events for timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float total_time_custom = 0.0f;
    
    // Benchmark custom kernel over many iterations
    for (int i = 0; i < ITERATIONS; i++) {
        cudaEventRecord(startEvent, 0);
        sumReductionOptimized<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, N);
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        float ms;
        cudaEventElapsedTime(&ms, startEvent, stopEvent);
        total_time_custom += ms;
    }
    float avg_time_custom = total_time_custom / ITERATIONS;  // in ms
    float custom_result = finalReductionOnCPU(d_output, blocks);

    // Setup cuBLAS and warm it up
    cublasHandle_t handle;
    cublasCreate(&handle);
    float dummy;
    cublasSasum(handle, N, d_input, 1, &dummy);
    cudaDeviceSynchronize();

    float total_time_cublas = 0.0f;
    float cublas_result = 0.0f;
    // Benchmark cuBLAS over many iterations
    for (int i = 0; i < ITERATIONS; i++) {
        cudaEventRecord(startEvent, 0);
        cublasSasum(handle, N, d_input, 1, &cublas_result);
        cudaDeviceSynchronize();
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        float ms;
        cudaEventElapsedTime(&ms, startEvent, stopEvent);
        total_time_cublas += ms;
    }
    float avg_time_cublas = total_time_cublas / ITERATIONS;  // in ms
    cublasDestroy(handle);
    
    // Clean up CUDA events
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    
    // Convert times to seconds
    double seconds_custom = avg_time_custom / 1000.0;
    double seconds_cublas = avg_time_cublas / 1000.0;
    
    // Calculate GFLOPS (for reduction, we consider N adds as N floating-point operations)
    double gflops_custom = (N / seconds_custom) / 1e9;
    double gflops_cublas = (N / seconds_cublas) / 1e9;
    
    std::cout << "Custom Sum Reduction Result: " << custom_result << std::endl;
    std::cout << "Custom Sum Reduction GFLOPS: " << gflops_custom << std::endl;
    std::cout << "cuBLAS Sum Reduction Result: " << cublas_result << std::endl;
    std::cout << "cuBLAS Sum Reduction GFLOPS: " << gflops_cublas << std::endl;
    
    // Compare results (using relative tolerance)
    if (fabs(custom_result - cublas_result) / fabs(cublas_result) < 1e-3) {
        double speedup_percentage = (gflops_custom / gflops_cublas) * 100;
        std::cout << "Performance Comparison: " << std::endl;
        std::cout << "cuBLAS Speed: 100%" << std::endl;
        std::cout << "Custom Reduction Speed: " << speedup_percentage << "%" << std::endl;
    } else {
        std::cout << "Error: Results do not match!" << std::endl;
    }
    
    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    
    return 0;
}

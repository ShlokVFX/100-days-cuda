#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
            file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

// CUDA Kernel for complex math operations
__global__ void vectorComplexCompute(const float* A, const float* B, float* C, int numElements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numElements) {
        float temp = A[i] * B[i];
        for (int j = 0; j < 100; ++j) { // Adding workload
            temp = sqrt(temp) + exp(A[i]) - log(B[i] + 1);
        }
        C[i] = temp;
    }
}

// CPU version for complex math operations
void vectorComplexComputeCPU(const float* A, const float* B, float* C, int numElements) {
    for (int i = 0; i < numElements; ++i) {
        float temp = A[i] * B[i];
        for (int j = 0; j < 100; ++j) { // Simulating heavy computation
            temp = sqrt(temp) + exp(A[i]) - log(B[i] + 1);
        }
        C[i] = temp;
    }
}

int main(void) {
    int numElements = 100000000;  // 100 million elements
    size_t size = numElements * sizeof(float);
    int halfElements = numElements / 2;

    // Host memory allocation
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    float* h_C_CPU = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Benchmark CPU Execution
    clock_t start_cpu = clock();
    vectorComplexComputeCPU(h_A, h_B, h_C_CPU, numElements);
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000.0; // Convert to ms

    // Device memory allocation
    float* d_A, * d_B, * d_C;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C, size));

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));

    // Start GPU timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));

    // Asynchronously copy first half of data using stream1
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_A, h_A, halfElements * sizeof(float), cudaMemcpyHostToDevice, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_B, h_B, halfElements * sizeof(float), cudaMemcpyHostToDevice, stream1));

    // Asynchronously copy second half of data using stream2
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_A + halfElements, h_A + halfElements, halfElements * sizeof(float), cudaMemcpyHostToDevice, stream2));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_B + halfElements, h_B + halfElements, halfElements * sizeof(float), cudaMemcpyHostToDevice, stream2));

    // Kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfElements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernels on separate streams
    vectorComplexCompute << <blocksPerGrid, threadsPerBlock, 0, stream1 >> > (d_A, d_B, d_C, halfElements);
    vectorComplexCompute << <blocksPerGrid, threadsPerBlock, 0, stream2 >> > (d_A + halfElements, d_B + halfElements, d_C + halfElements, halfElements);

    // Copy results back asynchronously
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_C, d_C, halfElements * sizeof(float), cudaMemcpyDeviceToHost, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_C + halfElements, d_C + halfElements, halfElements * sizeof(float), cudaMemcpyDeviceToHost, stream2));

    // Synchronize both streams
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));

    // Stop GPU timing
    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float gpu_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_time, start, stop)); // Time in milliseconds

    // Print first 10 results
    printf("\nVector Computation Results (First 10 Elements):\n");
    for (int i = 0; i < 10; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Verify results
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_C_CPU[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("\nTest PASSED\n");

    // Print Benchmark Results
    printf("\n=== Performance Comparison ===\n");
    printf("CPU Time: %.3f ms\n", cpu_time);
    printf("GPU Time (CUDA Streams): %.3f ms\n", gpu_time);
    printf("Speedup (CPU / GPU): %.2fx\n", cpu_time / gpu_time);

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_CPU);

    return 0;
}

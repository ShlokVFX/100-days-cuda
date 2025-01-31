#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 10000000  // Large size for benchmarking

// CUDA Kernel for Vector Addition
__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// CPU Function for Vector Addition
void vectorAddCPU(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    size_t size = N * sizeof(float);

    // Allocate memory on host
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C_cpu = (float*)malloc(size);
    float* h_C_gpu = (float*)malloc(size);

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    // ---------------- CPU Benchmark ----------------
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_A, h_B, h_C_cpu, N);
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_cpu = stop_cpu - start_cpu;
    std::cout << "CPU Time: " << duration_cpu.count() << " ms" << std::endl;

    // ---------------- GPU Benchmark ----------------

    // Allocate memory on device
    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Configure kernel execution
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Use CUDA events for precise timing
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    // Start recording time
    cudaEventRecord(start_gpu);
    vectorAdd << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, N);
    cudaEventRecord(stop_gpu);

    // Synchronize and measure GPU execution time
    cudaEventSynchronize(stop_gpu);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
    std::cout << "GPU Time: " << gpu_time << " ms" << std::endl;

    // Copy result back to host
    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    // ---------------- Print Results ----------------
    std::cout << "\nFirst 10 Vector Addition Results (CPU vs. GPU):\n";
    std::cout << "Index\tCPU\tGPU\n";
    for (int i = 0; i < 10; i++) {
        std::cout << i << "\t" << h_C_cpu[i] << "\t" << h_C_gpu[i] << "\n";
    }

    // Verify results
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_C_cpu[i] != h_C_gpu[i]) {
            correct = false;
            std::cout << "Mismatch at index " << i << ": CPU(" << h_C_cpu[i] << ") GPU(" << h_C_gpu[i] << ")\n";
            break;
        }
    }
    if (correct) {
        std::cout << "\nResults match!\n";
    }
    else {
        std::cout << "\nResults do not match!\n";
    }

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free CPU memory
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}

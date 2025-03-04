#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include "matmul.h"
#include <cuda_runtime.h>

// Helper to run a single test
void run_test(size_t M, size_t N, size_t K)
{
    // Host memory
    std::vector<float> h_A(M*K), h_B(K*N), h_C(M*N, 0.0f);

    // Initialize A and B with some random values
    for (size_t i = 0; i < M*K; ++i) {
        h_A[i] = static_cast<float>(rand() % 10);
    }
    for (size_t i = 0; i < K*N; ++i) {
        h_B[i] = static_cast<float>(rand() % 10);
    }

    // Device pointers
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M*K*sizeof(float));
    cudaMalloc((void**)&d_B, K*N*sizeof(float));
    cudaMalloc((void**)&d_C, M*N*sizeof(float));

    cudaMemcpy(d_A, h_A.data(), M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K*N*sizeof(float), cudaMemcpyHostToDevice);

    // Measure time with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    solution(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Compute GFLOPS = (2 * M * N * K) / (time_in_sec * 1e9)
    double ops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    double gflops = ops / (milliseconds / 1000.0) / 1e9;

    std::cout << M << "x" << N << ": "
              << milliseconds << " ms, "
              << gflops << " GFLOPS\n";

    // Copy back (optional if you just want to measure time)
    cudaMemcpy(h_C.data(), d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main()
{
    // Run a few large tests
    run_test(4096, 4096, 4096);
    run_test(6144, 6144, 6144);
    run_test(7168, 7168, 7168);
    run_test(8192, 8192, 8192);
    run_test(9216, 9216, 9216);

    return 0;
}

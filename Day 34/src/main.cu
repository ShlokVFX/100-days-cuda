#include <iostream>
#include <vector>
#include <cstdlib>
#include "matmul.h"

int main()
{
    // Example matrix dimensions
    size_t M = 4, K = 4, N = 4;

    // Host vectors
    std::vector<float> h_A(M*K), h_B(K*N), h_C(M*N, 0.0f);

    // Fill h_A and h_B with some values
    for (size_t i = 0; i < M*K; ++i) {
        h_A[i] = static_cast<float>(rand() % 10);
    }
    for (size_t i = 0; i < K*N; ++i) {
        h_B[i] = static_cast<float>(rand() % 10);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M*K*sizeof(float));
    cudaMalloc((void**)&d_B, K*N*sizeof(float));
    cudaMalloc((void**)&d_C, M*N*sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A.data(), M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K*N*sizeof(float), cudaMemcpyHostToDevice);

    // Call the optimized matmul solution
    solution(d_A, d_B, d_C, M, N, K);

    // Copy the result back to the host
    cudaMemcpy(h_C.data(), d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    // Print out the resulting matrix
    std::cout << "Resulting C (4x4):\n";
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            std::cout << h_C[i*N + j] << " ";
        }
        std::cout << "\n";
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

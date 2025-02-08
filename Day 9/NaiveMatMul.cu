// naive_matmul.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 1024  // matrix dimension: N x N

__global__ void matMulNaive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    int size = N * N;
    size_t bytes = size * sizeof(float);

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    // Initialize matrices (for simplicity, A filled with 1's, B with 2's)
    for (int i = 0; i < size; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    matMulNaive << <grid, block >> > (d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify: each element should equal 1*2*N = 2*N.
    bool correct = true;
    for (int i = 0; i < size; i++) {
        float expected = 2.0f * N;
        if (fabs(h_C[i] - expected) > 1e-5) {
            correct = false;
            printf("Mismatch at index %d: got %f, expected %f\n", i, h_C[i], expected);
            break;
        }
    }
    printf("Naive multiplication %s\n", correct ? "successful!" : "FAILED");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}

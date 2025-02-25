#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 3 // 3x3 matrix

// CUDA Kernel for Matrix Multiplication
__global__ void Matmul(int* A, int* B, int* C, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col]; // Correct dot product
        }
        C[row * n + col] = sum;
    }
}

int main() {
    // Memory size calculation
    int size = N * N * sizeof(int);

    // Host matrices
    int h_A[N * N], h_B[N * N], h_C[N * N];

    // Initialize host matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = i + 1;           // Example: 1, 2, 3, ..., 9
        h_B[i] = (i + 1) * 2;     // Example: 2, 4, 6, ..., 18
    }

    // Print Input Matrices
    printf("Matrix A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_A[i * N + j]);
        }
        printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_B[i * N + j]);
        }
        printf("\n");
    }

    // Device matrices
    int *d_A, *d_B, *d_C;

    // Allocate memory on GPU
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from Host to Device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 BlockDim(16, 16); // 16x16 threads per block
    dim3 GridDim((N + 15) / 16, (N + 15) / 16); // Grid size

    // Launch kernel
    Matmul<<<GridDim, BlockDim>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print result
    printf("\nMatrix C (Result):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

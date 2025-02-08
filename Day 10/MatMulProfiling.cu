#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 16  // Tile size for shared memory optimization

// CUDA Kernel for Matrix Multiplication using Tiling
__global__ void matrixMulShared(float* C, float* A, float* B, int N) {
    __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (N / BLOCK_SIZE); ++t) {
        Asub[ty][tx] = A[row * N + (t * BLOCK_SIZE + tx)];
        Bsub[ty][tx] = B[(t * BLOCK_SIZE + ty) * N + col];

        __syncthreads();  // Sync threads in block

        for (int k = 0; k < BLOCK_SIZE; ++k)
            sum += Asub[ty][k] * Bsub[k][tx];

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }

    // Print first few thread results for debugging
    if (row == 0 && col < 5) {
        printf("Thread (0, %d): C[%d] = %f\n", col, row * N + col, sum);
    }
}

// Initialize matrices with random values
void initializeMatrix(float* mat, int N) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = static_cast<float>(rand() % 100) / 10.0f;
    }
}

// Print matrix for debugging (small matrices only)
void printMatrix(float* mat, int N) {
    for (int i = 0; i < 5; i++) {  // Print first 5 rows
        for (int j = 0; j < 5; j++) {  // Print first 5 columns
            printf("%6.2f ", mat[i * N + j]);
        }
        printf("\n");
    }
    printf("...\n");  // Indicate truncation for large matrices
}

int main() {
    const int N = 1024;  // Matrix size
    size_t bytes = N * N * sizeof(float);

    printf("Allocating memory...\n");

    // Allocate memory
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    initializeMatrix(h_A, N);
    initializeMatrix(h_B, N);

    printf("Matrices initialized.\n");

    // Allocate device memory
    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    printf("Copying data to GPU...\n");

    // Copy data to GPU
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(N / BLOCK_SIZE, N / BLOCK_SIZE);

    printf("Launching CUDA Kernel...\n");
    matrixMulShared << <blocks, threads >> > (d_C, d_A, d_B, N);

    cudaDeviceSynchronize();  // Ensure kernel execution is finished

    printf("Kernel execution completed. Copying results back to CPU...\n");

    // Copy result back
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    printf("Computation done! Printing a small portion of result matrix:\n");
    printMatrix(h_C, N);

    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    printf("Memory freed. Exiting program.\n");

    return 0;
}

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>

#ifndef DIM
#define DIM 8192/16
#endif

#define BLOCK_SIZE 32
#define R 23
#define M (DIM / BLOCK_SIZE)

__global__ void naiveMatMulKernel(const float* A, const float* B, float* C, int dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < dim && col < dim) {
        float sum = 0.0f;
        for (int k = 0; k < dim; k++) {
            sum += A[row * dim + k] * B[k * dim + col];
        }
        C[row * dim + col] = sum;
    }
}

__global__ void alphaTensorLargeMatMulKernel(const float* A, const float* B, float* C, 
                                             const float* U, const float* V, const float* W) {
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row >= DIM || col >= DIM) return;
    
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];
    float sum = 0.0f;
    for (int k = 0; k < M; k++) {
        s_A[threadIdx.y][threadIdx.x] = A[row * DIM + k * BLOCK_SIZE + threadIdx.x];
        s_B[threadIdx.y][threadIdx.x] = B[(k * BLOCK_SIZE + threadIdx.y) * DIM + col];
        __syncthreads();
        for (int r = 0; r < R; r++) {
            sum += U[threadIdx.y * R + r] * s_A[threadIdx.y][threadIdx.x] * V[threadIdx.x * R + r] * s_B[threadIdx.y][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * DIM + col] = sum;
}

int main() {
    size_t bytes = DIM * DIM * sizeof(float);
    float *A, *B, *C_baseline, *C_alphatensor;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C_baseline, bytes);
    cudaMallocManaged(&C_alphatensor, bytes);
    for (int i = 0; i < DIM * DIM; i++) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(DIM / BLOCK_SIZE, DIM / BLOCK_SIZE);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    naiveMatMulKernel<<<gridDim, blockDim>>>(A, B, C_baseline, DIM);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float millisecondsBaseline;
    cudaEventElapsedTime(&millisecondsBaseline, start, stop);

    size_t sizeU = BLOCK_SIZE * R * sizeof(float);
    size_t sizeW = BLOCK_SIZE * BLOCK_SIZE * R * sizeof(float);
    float *U, *V, *W;
    cudaMallocManaged(&U, sizeU);
    cudaMallocManaged(&V, sizeU);
    cudaMallocManaged(&W, sizeW);
    for (int i = 0; i < BLOCK_SIZE * R; i++) { U[i] = 1.0f; V[i] = 1.0f; }
    for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE * R; i++) { W[i] = 1.0f; }
    cudaEventRecord(start);
    alphaTensorLargeMatMulKernel<<<gridDim, blockDim>>>(A, B, C_alphatensor, U, V, W);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float millisecondsAlphaTensor;
    cudaEventElapsedTime(&millisecondsAlphaTensor, start, stop);
    float speedup = ((millisecondsBaseline - millisecondsAlphaTensor) / millisecondsBaseline) * 100.0f;
    std::cout << "Naive GEMM baseline time: " << millisecondsBaseline << " ms" << std::endl;
    std::cout << "AlphaTensor GPU-optimized time: " << millisecondsAlphaTensor << " ms" << std::endl;
    std::cout << "AlphaTensor GPU-optimized vs naive GEMM: " << speedup << "% speedup" << std::endl;
    cudaFree(A);
    cudaFree(B);
    cudaFree(C_baseline);
    cudaFree(C_alphatensor);
    cudaFree(U);
    cudaFree(V);
    cudaFree(W);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

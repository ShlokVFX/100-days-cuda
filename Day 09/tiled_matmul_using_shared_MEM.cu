// tiled_matmul.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 1024
#define TILE_WIDTH 16

__global__ void matMulTiled(const float* A, const float* B, float* C, int n) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (n + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < numTiles; t++) {
        int tiledCol = t * TILE_WIDTH + threadIdx.x;
        int tiledRow = t * TILE_WIDTH + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] = (row < n && tiledCol < n) ? A[row * n + tiledCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (tiledRow < n && col < n) ? B[tiledRow * n + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < n && col < n)
        C[row * n + col] = sum;
}

int main() {
    int size = N * N;
    size_t bytes = size * sizeof(float);

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

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

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);
    matMulTiled << <grid, block >> > (d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < size; i++) {
        float expected = 2.0f * N;
        if (fabs(h_C[i] - expected) > 1e-5) {
            correct = false;
            printf("Mismatch at index %d: got %f, expected %f\n", i, h_C[i], expected);
            break;
        }
    }
    printf("Tiled multiplication %s\n", correct ? "successful!" : "FAILED");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}

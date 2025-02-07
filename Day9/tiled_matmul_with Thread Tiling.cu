// thread_tiled_matmul.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 1024
#define TILE_WIDTH 16
#define THREAD_TILE 2  // each thread computes 2 output elements (along columns)

__global__ void matMulTiledThreadTile(const float* A, const float* B, float* C, int n) {
    // Block covers a TILE_WIDTH x TILE_WIDTH tile of C.
    // We set blockDim.x = TILE_WIDTH/THREAD_TILE and blockDim.y = TILE_WIDTH.
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x * THREAD_TILE;

    float sum[THREAD_TILE] = { 0.0f, 0.0f };
    int numTiles = (n + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++) {
        // Each thread loads one element of A into shared memory.
        int aCol = t * TILE_WIDTH + threadIdx.x * THREAD_TILE;
        for (int i = 0; i < THREAD_TILE; i++) {
            int colA = aCol + i;
            tileA[threadIdx.y][threadIdx.x * THREAD_TILE + i] =
                (row < n && colA < n) ? A[row * n + colA] : 0.0f;
        }
        // Each thread loads one element of B into shared memory.
        int bRow = t * TILE_WIDTH + threadIdx.y;
        int bCol = blockIdx.x * TILE_WIDTH + threadIdx.x * THREAD_TILE;
        for (int i = 0; i < THREAD_TILE; i++) {
            int colB = bCol + i;
            tileB[threadIdx.y][threadIdx.x * THREAD_TILE + i] =
                (bRow < n && colB < n) ? B[bRow * n + colB] : 0.0f;
        }
        __syncthreads();

        // Compute partial products from the current tile.
        for (int k = 0; k < TILE_WIDTH; k++) {
            float aVal = tileA[threadIdx.y][k];
            for (int i = 0; i < THREAD_TILE; i++) {
                sum[i] += aVal * tileB[k][threadIdx.x * THREAD_TILE + i];
            }
        }
        __syncthreads();
    }

    // Write computed tile to global memory.
    for (int i = 0; i < THREAD_TILE; i++) {
        int cCol = col + i;
        if (row < n && cCol < n)
            C[row * n + cCol] = sum[i];
    }
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

    // Set block dimensions: 
    // blockDim.x = TILE_WIDTH/THREAD_TILE, blockDim.y = TILE_WIDTH.
    dim3 block(TILE_WIDTH / THREAD_TILE, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    matMulTiledThreadTile << <grid, block >> > (d_A, d_B, d_C, N);
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
    printf("Tiled thread-tiled multiplication %s\n", correct ? "successful!" : "FAILED");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}

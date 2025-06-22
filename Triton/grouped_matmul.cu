#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define GROUP_SIZE 3

__global__ void matmul_naive(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i)
            sum += A[row * K + i] * B[i * N + col];
        C[row * N + col] = sum;
    }
}

__global__ void matmul_row_major(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        if (row < M && tile * TILE_SIZE + threadIdx.x < K)
            Asub[threadIdx.y][threadIdx.x] = A[row * K + tile * TILE_SIZE + threadIdx.x];
        else
            Asub[threadIdx.y][threadIdx.x] = 0.0f;

        if (tile * TILE_SIZE + threadIdx.y < K && col < N)
            Bsub[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bsub[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
            sum += Asub[threadIdx.y][i] * Bsub[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

__global__ void matmul_grouped(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

    int group_id = blockIdx.x;
    int tile_id_in_group = threadIdx.z;

    int tiles_per_row = N / TILE_SIZE;
    int tile_row = (group_id / tiles_per_row) * GROUP_SIZE + tile_id_in_group;
    int tile_col = group_id % tiles_per_row;

    int row = tile_row * TILE_SIZE + threadIdx.y;
    int col = tile_col * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int tiled_k = tile * TILE_SIZE;

        if (row < M && tiled_k + threadIdx.x < K)
            Asub[threadIdx.y][threadIdx.x] = A[row * K + tiled_k + threadIdx.x];
        else
            Asub[threadIdx.y][threadIdx.x] = 0.0f;

        if (tiled_k + threadIdx.y < K && col < N)
            Bsub[threadIdx.y][threadIdx.x] = B[(tiled_k + threadIdx.y) * N + col];
        else
            Bsub[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i)
            sum += Asub[threadIdx.y][i] * Bsub[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

void launch_and_time(void (*kernel)(float*, float*, float*, int, int, int), float* A, float* B, float* C, int M, int N, int K, dim3 blocks, dim3 threads, const char* name) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << name << " execution time: " << ms << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int M = 512, N = 512, K = 512;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *A, *B, *C;
    cudaMallocManaged(&A, size_A);
    cudaMallocManaged(&B, size_B);
    cudaMallocManaged(&C, size_C);

    for (int i = 0; i < M * K; ++i) A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) B[i] = 1.0f;

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    launch_and_time(matmul_naive, A, B, C, M, N, K, blocks, threads, "Naive Matmul");
    launch_and_time(matmul_row_major, A, B, C, M, N, K, blocks, threads, "Row-Major Matmul");

    dim3 group_threads(TILE_SIZE, TILE_SIZE, GROUP_SIZE);
    int num_groups = (M / TILE_SIZE / GROUP_SIZE) * (N / TILE_SIZE);
    launch_and_time(matmul_grouped, A, B, C, M, N, K, dim3(num_groups), group_threads, "Grouped Matmul");

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}

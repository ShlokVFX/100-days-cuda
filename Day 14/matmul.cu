#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define N 1024  // Matrix size (1024x1024)
#define TILE 16 // Tiling size

// -------------------- Naïve Kernel -------------------- //
__global__ void naive_matmul(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// -------------------- Warp-Tiling with Tensor Cores -------------------- //
#include <mma.h>
using namespace nvcuda;

#define WARP_SIZE 32
#define M 16
#define K 16
#define N_T 16

__global__ void tensorcore_matmul(half *A, half *B, float *C, int n) {
    int warpM = blockIdx.y * (WARP_SIZE / M);
    int warpN = blockIdx.x * (WARP_SIZE / N_T);

    wmma::fragment<wmma::matrix_a, M, N_T, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N_T, K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N_T, K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < n; k += K) {
        wmma::load_matrix_sync(a_frag, A + warpM * n + k, n);
        wmma::load_matrix_sync(b_frag, B + k * n + warpN, n);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warpM * n + warpN, c_frag, n, wmma::mem_row_major);
}

// -------------------- Main Function -------------------- //
int main() {
    size_t bytes = N * N * sizeof(float);
    size_t bytes_half = N * N * sizeof(half);

    float *h_A, *h_B, *h_C_naive, *h_C_tc;
    cudaMallocHost(&h_A, bytes);
    cudaMallocHost(&h_B, bytes);
    cudaMallocHost(&h_C_naive, bytes);
    cudaMallocHost(&h_C_tc, bytes);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_A, *d_B, *d_C_naive;
    half *d_A_half, *d_B_half;
    float *d_C_tc;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C_naive, bytes);
    cudaMalloc(&d_A_half, bytes_half);
    cudaMalloc(&d_B_half, bytes_half);
    cudaMalloc(&d_C_tc, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE, TILE);
    dim3 gridDim(N / TILE, N / TILE);

    cudaEvent_t start, stop;
    float naiveTime, tcTime;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ------------------ Measure Naïve MatMul ------------------ //
    cudaEventRecord(start);
    naive_matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C_naive, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&naiveTime, start, stop);

    cudaMemcpy(h_C_naive, d_C_naive, bytes, cudaMemcpyDeviceToHost);

    // ------------------ Measure Tensor Core MatMul ------------------ //
    cudaMemcpy(d_A_half, h_A, bytes_half, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_half, h_B, bytes_half, cudaMemcpyHostToDevice);

    dim3 gridDimTC(N / WARP_SIZE, N / WARP_SIZE);

    cudaEventRecord(start);
    tensorcore_matmul<<<gridDimTC, WARP_SIZE>>>(d_A_half, d_B_half, d_C_tc, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tcTime, start, stop);

    cudaMemcpy(h_C_tc, d_C_tc, bytes, cudaMemcpyDeviceToHost);

    // ------------------ Compute GFLOPS ------------------ //
    float naiveGFLOPS = (2.0 * N * N * N) / (naiveTime * 1e6);
    float tcGFLOPS = (2.0 * N * N * N) / (tcTime * 1e6);
    float speedup = ((tcGFLOPS - naiveGFLOPS) / naiveGFLOPS) * 100;

    std::cout << "Naive MatMul GFLOPS: " << naiveGFLOPS << std::endl;
    std::cout << "Tensor Core MatMul GFLOPS: " << tcGFLOPS << std::endl;
    std::cout << "Speedup: " << speedup << "% faster" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_naive);
    cudaFree(d_A_half);
    cudaFree(d_B_half);
    cudaFree(d_C_tc);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C_naive);
    cudaFreeHost(h_C_tc);

    return 0;
}

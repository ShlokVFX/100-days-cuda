#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Reduced block sizes to avoid register pressure
const int BM = 64;   // BLOCK_M
const int BN = 64;   // BLOCK_N
const int BK = 8;    // BLOCK_K

template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void optimized_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    // Shared memory for double buffering
    __shared__ float smem_A[2][64][8];  // Using actual numbers for shared memory
    __shared__ float smem_B[2][8][64];

    // Thread block coordinates
    const int block_x = blockIdx.x * BLOCK_N;
    const int block_y = blockIdx.y * BLOCK_M;

    // Thread coordinates within block
    const int thread_x = threadIdx.x;
    const int thread_y = threadIdx.y;

    // Global coordinates
    const int row = block_y + thread_y;
    const int col = block_x + thread_x;

    // Initialize accumulator
    float acc = 0.0f;

    // Loop over K dimension
    for (int k = 0; k < K; k += BLOCK_K) {
        // Load data into shared memory
        if (row < M && (k + thread_x) < K && thread_x < BLOCK_K) {
            smem_A[0][thread_y][thread_x] = A[row * K + k + thread_x];
        }
        else {
            smem_A[0][thread_y][thread_x] = 0.0f;
        }

        if (col < N && (k + thread_y) < K && thread_y < BLOCK_K) {
            smem_B[0][thread_y][thread_x] = B[(k + thread_y) * N + col];
        }
        else {
            smem_B[0][thread_y][thread_x] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum
#pragma unroll
        for (int i = 0; i < BLOCK_K; ++i) {
            acc += smem_A[0][thread_y][i] * smem_B[0][i][thread_x];
        }

        __syncthreads();
    }

    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// CPU version for result verification
void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Utility function to check for CUDA errors
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
            file, line, static_cast<unsigned int>(err),
            cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Matrix dimensions - reduced for testing
    int M = 512;  // A rows
    int N = 512;  // B columns
    int K = 512;  // A columns and B rows

    // Allocate host memory
    float* h_A = (float*)malloc(M * K * sizeof(float));
    float* h_B = (float*)malloc(K * N * sizeof(float));
    float* h_C = (float*)malloc(M * N * sizeof(float));
    float* h_C_ref = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices with small values
    for (int i = 0; i < M * K; i++) {
        h_A[i] = (float)(rand() % 10) / 10.0f;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = (float)(rand() % 10) / 10.0f;
    }

    // Allocate device memory
    float* d_A, * d_B, * d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, M * N * sizeof(float)));

    // Copy inputs to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Launch configuration
    dim3 threadsPerBlock(16, 16);  // 256 threads per block
    dim3 numBlocks(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    // Clear any previous errors
    cudaGetLastError();

    // Warm up run
    optimized_gemm_kernel<BM, BN, BK>
        << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, M, N, K);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Timing run
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    optimized_gemm_kernel<BM, BN, BK>
        << <numBlocks, threadsPerBlock >> > (d_A, d_B, d_C, M, N, K);

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute reference result on CPU
    cpu_gemm(h_A, h_B, h_C_ref, M, N, K);

    // Verify result
    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float error = fabs(h_C[i] - h_C_ref[i]);
        max_error = max(max_error, error);
    }

    // Calculate performance
    float gflops = (2.0f * M * N * K) / (milliseconds * 1e6);

    printf("Matrix multiplication results:\n");
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("Max error: %e\n", max_error);
    printf("Time: %.2f ms\n", milliseconds);

    // Cleanup
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return 0;
}
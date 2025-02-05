Optimized CUDA GEMM (General Matrix Multiply)

This repository contains a CUDA-based implementation of a matrix multiplication (GEMM) kernel that leverages shared memory for improved performance. A CPU version is also provided for result verification and performance comparison.
Table of Contents

    Overview
    Code Structure and Components
        Constants and Block Sizes
        The Optimized CUDA Kernel
        CPU Reference Implementation
        CUDA Error Checking Utility
        Main Function
    Compilation and Execution
    Performance and Verification
    License

Overview

The code implements an optimized version of the GEMM operation using CUDA. The main highlights include:

    Block Tiling and Shared Memory Usage: The kernel partitions the matrices into smaller tiles (blocks) to maximize data reuse from shared memory.
    Double Buffering Setup (Partial): Although double buffering is declared in shared memory arrays, the sample uses only one buffer index. This structure can be extended to overlap computation with memory loads.
    Verification Against CPU: A standard CPU-based GEMM is used to verify the correctness of the GPU implementation.
    Performance Measurement: CUDA events are used to time the kernel execution, and the performance is reported in GFLOPS.

Code Structure and Components
Constants and Block Sizes

// Reduced block sizes to avoid register pressure
const int BM = 64;   // BLOCK_M: Number of rows per block from matrix A and C
const int BN = 64;   // BLOCK_N: Number of columns per block from matrix B and C
const int BK = 8;    // BLOCK_K: The inner dimension size processed per iteration

    BM, BN, BK: These constants define the tile (block) dimensions used for the matrix multiplication. Choosing smaller block sizes can help reduce register pressure and shared memory usage.

The Optimized CUDA Kernel

template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void optimized_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M, const int N, const int K
) {
    // Shared memory for double buffering
    __shared__ float smem_A[2][64][8];  // Two buffers for matrix A tiles
    __shared__ float smem_B[2][8][64];    // Two buffers for matrix B tiles

    // Compute block starting coordinates in the global matrix
    const int block_x = blockIdx.x * BLOCK_N;
    const int block_y = blockIdx.y * BLOCK_M;

    // Thread coordinates within the block
    const int thread_x = threadIdx.x;
    const int thread_y = threadIdx.y;

    // Global matrix coordinates for the output element
    const int row = block_y + thread_y;
    const int col = block_x + thread_x;

    // Initialize accumulator for the output element
    float acc = 0.0f;

    // Loop over the K-dimension in tiles of size BLOCK_K
    for (int k = 0; k < K; k += BLOCK_K) {
        // Load a tile of matrix A into shared memory.
        if (row < M && (k + thread_x) < K && thread_x < BLOCK_K) {
            smem_A[0][thread_y][thread_x] = A[row * K + k + thread_x];
        } else {
            smem_A[0][thread_y][thread_x] = 0.0f;
        }

        // Load a tile of matrix B into shared memory.
        if (col < N && (k + thread_y) < K && thread_y < BLOCK_K) {
            smem_B[0][thread_y][thread_x] = B[(k + thread_y) * N + col];
        } else {
            smem_B[0][thread_y][thread_x] = 0.0f;
        }

        // Synchronize to ensure all threads have loaded their data.
        __syncthreads();

        // Compute the partial sum for this tile.
#pragma unroll
        for (int i = 0; i < BLOCK_K; ++i) {
            acc += smem_A[0][thread_y][i] * smem_B[0][i][thread_x];
        }

        // Synchronize before loading the next tile.
        __syncthreads();
    }

    // Write the computed value to global memory if within bounds.
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

Key Points:

    Template Parameters: The kernel uses template parameters (BLOCK_M, BLOCK_N, BLOCK_K) so that block sizes can be adjusted at compile time.
    Shared Memory: Two 3D arrays (smem_A and smem_B) are declared for double buffering (currently using the first buffer only). These arrays reduce global memory accesses by reusing data across multiple threads.
    Thread and Block Indices:
        block_x and block_y compute the starting positions in the global matrix for each block.
        thread_x and thread_y provide each thread’s coordinates within the block.
    Tiled Loop: The outer loop iterates over the K dimension in increments of BLOCK_K. Each tile is loaded into shared memory, and a nested loop performs partial dot-product computations.
    Synchronization: __syncthreads() is used to ensure that all threads complete their shared memory loads and computations before moving on.
    Result Storage: After accumulating the sum, the result is stored back into the global memory array C.

CPU Reference Implementation

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

    Purpose: This function implements the straightforward triple-nested loop for matrix multiplication. It serves as a correctness reference for the CUDA kernel.
    Usage: After the GPU computation, the CPU result is used to verify the accuracy of the GPU result by comparing each element.

CUDA Error Checking Utility

// Utility macro and function to check for CUDA errors.
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

    Error Checking Macro: CHECK_CUDA_ERROR wraps CUDA API calls to check the return value.
    Error Reporting: If an error occurs, the function prints the file name, line number, error code, and error string before exiting. This is essential for debugging CUDA applications.

Main Function

int main() {
    // Matrix dimensions - reduced for testing
    int M = 512;  // Number of rows in matrix A and C
    int N = 512;  // Number of columns in matrix B and C
    int K = 512;  // Shared dimension between A and B

    // Allocate host memory
    float* h_A = (float*)malloc(M * K * sizeof(float));
    float* h_B = (float*)malloc(K * N * sizeof(float));
    float* h_C = (float*)malloc(M * N * sizeof(float));
    float* h_C_ref = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices with small random values
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

    // Copy host data to device memory
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Configure the kernel launch parameters
    dim3 threadsPerBlock(16, 16);  // 256 threads per block
    dim3 numBlocks(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    // Clear any previous CUDA errors
    cudaGetLastError();

    // Warm-up kernel launch (helps avoid cold-start overhead)
    optimized_gemm_kernel<BM, BN, BK>
        <<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Timing the kernel execution using CUDA events
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    optimized_gemm_kernel<BM, BN, BK>
        <<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    // Calculate the elapsed time in milliseconds
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy the result back from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute the reference result on the CPU for validation
    cpu_gemm(h_A, h_B, h_C_ref, M, N, K);

    // Verify the result by comparing each element and record the maximum error
    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float error = fabs(h_C[i] - h_C_ref[i]);
        max_error = max(max_error, error);
    }

    // Calculate performance in GFLOPS
    float gflops = (2.0f * M * N * K) / (milliseconds * 1e6);

    // Print the performance and error information
    printf("Matrix multiplication results:\n");
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("Max error: %e\n", max_error);
    printf("Time: %.2f ms\n", milliseconds);

    // Cleanup: destroy events and free device and host memory
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

Explanation:

    Memory Allocation and Initialization:
        Host Memory: Allocates memory for matrices h_A, h_B, h_C (result from GPU), and h_C_ref (CPU reference result).
        Matrix Initialization: Random values (scaled down) are assigned to matrices A and B.

    Device Memory Management:
        Allocation: Device memory for matrices is allocated using cudaMalloc.
        Data Transfer: Host matrices are copied to device memory using cudaMemcpy.

    Kernel Launch and Timing:
        Kernel Configuration: The grid is configured with blocks of size BM x BN and each block uses 16 x 16 threads.
        Warm-up Run: A warm-up launch minimizes the initial overhead.
        Timing Run: CUDA events (start and stop) measure the kernel execution time.

    Result Verification and Performance Measurement:
        Copy Back and Verification: The result from the GPU is copied back to host memory and compared against the CPU result.
        GFLOPS Calculation: The performance is computed based on the total number of floating-point operations and the elapsed time.

    Cleanup: All allocated resources (CUDA events, device memory, and host memory) are properly released.

    Performance and Verification

    Performance Metrics: The kernel’s performance is reported in GFLOPS.
    Verification: The CPU implementation of GEMM serves as a reference to ensure the GPU implementation is correct. The maximum error printed should be within acceptable numerical precision limits.

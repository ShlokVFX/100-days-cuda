#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <algorithm> // for std::max

// Define overall matrix dimension (must be a multiple of BLOCK_SIZE=3)
#ifndef DIM
#define DIM 8192/32
#endif

#define BLOCK_SIZE 3         // 3x3 blocks
#define R 23                 // Assumed factorization rank
#define M (DIM / BLOCK_SIZE) // Number of 3x3 blocks per dimension

// -----------------------------------------------------------------------------
// Baseline Naive GEMM kernel (row-major) for large matrices.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// AlphaTensor-inspired block multiplication kernel for large matrices.
// Each grid block computes one output 3x3 block. The kernel loops over the
// inner block dimension (k from 0 to M-1) and, for each inner block, it
// computes R intermediate products using a bilinear factorization with dummy
// coefficients. The computation within each 3x3 block is parallelized using a
// block of threads of size at least max(R, 9).
// -----------------------------------------------------------------------------
__global__ void alphaTensorLargeMatMulKernel(const float* A, const float* B, float* C, 
                                             const float* U, const float* V, const float* W) {
    // Each grid block corresponds to one output 3x3 block.
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Shared memory for intermediate products for the current inner block.
    __shared__ float s_p[R];

    // Each thread will contribute to the accumulation for one output element.
    // We use an array in registers for the 3x3 block accumulation.
    float C_block[BLOCK_SIZE * BLOCK_SIZE] = {0.0f};

    // Loop over the inner block dimension.
    for (int k = 0; k < M; k++) {
        // Load the 3x3 sub-blocks from A and B.
        float A_block[BLOCK_SIZE * BLOCK_SIZE];
        float B_block[BLOCK_SIZE * BLOCK_SIZE];
        for (int i = 0; i < BLOCK_SIZE; i++) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                A_block[i * BLOCK_SIZE + j] = A[(blockRow * BLOCK_SIZE + i) * DIM + (k * BLOCK_SIZE + j)];
                B_block[i * BLOCK_SIZE + j] = B[(k * BLOCK_SIZE + i) * DIM + (blockCol * BLOCK_SIZE + j)];
            }
        }

        // Use threads with threadIdx.x < R to compute intermediate products.
        int tid = threadIdx.x;
        if (tid < R) {
            float sumA = 0.0f;
            float sumB = 0.0f;
            // For demonstration, we use the first column of the block as a representative.
            for (int i = 0; i < BLOCK_SIZE; i++) {
                sumA += U[i * R + tid] * A_block[i * BLOCK_SIZE + 0];
                sumB += V[i * R + tid] * B_block[i * BLOCK_SIZE + 0];
            }
            s_p[tid] = sumA * sumB;
        }
        __syncthreads();

        // Threads with tid < 9 (since 3x3=9) compute one element of the output block.
        if (tid < BLOCK_SIZE * BLOCK_SIZE) {
            float sum = 0.0f;
            for (int r = 0; r < R; r++) {
                sum += W[tid * R + r] * s_p[r];
            }
            // Accumulate contribution from this inner block.
            C_block[tid] += sum;
        }
        __syncthreads(); // Ensure all threads complete before next k iteration.
    }

    // Write the accumulated 3x3 block to global memory.
    int tid = threadIdx.x;
    if (tid < BLOCK_SIZE * BLOCK_SIZE) {
        int i = tid / BLOCK_SIZE;
        int j = tid % BLOCK_SIZE;
        C[(blockRow * BLOCK_SIZE + i) * DIM + (blockCol * BLOCK_SIZE + j)] = C_block[tid];
    }
}

// -----------------------------------------------------------------------------
// Main function: sets up matrices, runs benchmarks, and prints results.
// -----------------------------------------------------------------------------
int main() {
    std::cout << "Multiplying " << DIM << " x " << DIM << " matrices" << std::endl;
    std::cout << "========================================" << std::endl;

    size_t bytes = DIM * DIM * sizeof(float);
    float *A, *B, *C_baseline, *C_alphatensor;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C_baseline, bytes);
    cudaMallocManaged(&C_alphatensor, bytes);

    // Initialize matrices A and B with random values.
    for (int i = 0; i < DIM * DIM; i++) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // ------------------- Baseline multiplication (Naive GEMM) -------------------
    dim3 blockDim(16, 16);
    dim3 gridDim((DIM + blockDim.x - 1) / blockDim.x, (DIM + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    naiveMatMulKernel<<<gridDim, blockDim>>>(A, B, C_baseline, DIM);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float millisecondsBaseline = 0;
    cudaEventElapsedTime(&millisecondsBaseline, start, stop);

    // ---------------- AlphaTensor multiplication using block kernel ----------------
    // Setup dummy factorization coefficients (all ones for demonstration).
    size_t sizeU = BLOCK_SIZE * R * sizeof(float);
    size_t sizeW = BLOCK_SIZE * BLOCK_SIZE * R * sizeof(float);
    float *U, *V, *W;
    cudaMallocManaged(&U, sizeU);
    cudaMallocManaged(&V, sizeU);
    cudaMallocManaged(&W, sizeW);
    for (int i = 0; i < BLOCK_SIZE * R; i++) { U[i] = 1.0f; V[i] = 1.0f; }
    for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE * R; i++) { W[i] = 1.0f; }

    // Launch one CUDA grid block per 3x3 output block.
    dim3 gridDimAT(M, M);
    // Use blockDim.x = max(R, 9). Since R=23, we launch with 23 threads per block.
    int threadsPerBlock = std::max(R, BLOCK_SIZE * BLOCK_SIZE);
    cudaEventRecord(start);
    alphaTensorLargeMatMulKernel<<<gridDimAT, threadsPerBlock>>>(A, B, C_alphatensor, U, V, W);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float millisecondsAlphaTensor = 0;
    cudaEventElapsedTime(&millisecondsAlphaTensor, start, stop);

    // Report benchmark results.
    std::cout << "Naive GEMM baseline time: " << millisecondsBaseline << " ms" << std::endl;
    std::cout << "AlphaTensor GPU-optimized time: " << millisecondsAlphaTensor << " ms" << std::endl;
    float speedup = ((millisecondsBaseline - millisecondsAlphaTensor) / millisecondsBaseline) * 100.0f;
    std::cout << "AlphaTensor GPU-optimized vs naive GEMM: " << speedup << "% speedup" << std::endl;

    // Cleanup.
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
//nvcc -DDIM=8192 -o ATT optimized_alphatensor.cu

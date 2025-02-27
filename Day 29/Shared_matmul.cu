#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

// #include <cuda_runtime.h>

// Note: input_a, input_b, and output_c are all device pointers to float arrays
//extern "C" void solution(float* input_a, float* input_b, float* output_c, size_t m, size_t n, size_t k) {
    
//} 

inline void checkCudaErrors(cudaError_t err, const char* msg = "")
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << msg << " -> "
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void solution(const float* input_a, const float* input_b, float* output_c,
    int M, int K, int N)
{
const int TILE_WIDTH = 32;

__shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
__shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

int row = blockDim.y * blockIdx.y + threadIdx.y;
int col = blockDim.x * blockIdx.x + threadIdx.x;

float sum = 0.0f;

int tx = threadIdx.x;
int ty = threadIdx.y;

for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++)
{
int tiledCol = t * TILE_WIDTH + tx;
if (row < M && tiledCol < K)
sharedA[ty][tx] = input_a[row * K + tiledCol];
else
sharedA[ty][tx] = 0.0f;
int tiledRow = t * TILE_WIDTH + ty;
if (tiledRow < K && col < N)
sharedB[ty][tx] = input_b[tiledRow * N + col];
else
sharedB[ty][tx] = 0.0f;

__syncthreads();

#pragma unroll
for (int i = 0; i < TILE_WIDTH; i++)
{
sum += sharedA[ty][i] * sharedB[i][tx];
}
__syncthreads();
}
if (row < M && col < N)
{
output_c[row * N + col] = sum;
}
}


int main()
{
    // Dimensions of matrices
    size_t M = 1024/256;  // rows in A and C
    size_t K = 1024/256;  // cols in A, rows in B
    size_t N = 1024/256;  // cols in B and C

    std::cout << "Matrix multiplication: A(" << M << "x" << K 
              << ") * B(" << K << "x" << N << ") = C(" 
              << M << "x" << N << ")\n";

    float *h_A, *h_B, *h_C;
    checkCudaErrors(cudaMallocHost(&h_A,       M * K * sizeof(float)), "malloc pinned h_A");
    checkCudaErrors(cudaMallocHost(&h_B,       K * N * sizeof(float)), "malloc pinned h_B");
    checkCudaErrors(cudaMallocHost(&h_C,       M * N * sizeof(float)), "malloc pinned h_C");

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            h_A[i * K + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_B[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_A, M * K * sizeof(float)), "malloc d_A");
    checkCudaErrors(cudaMalloc((void**)&d_B, K * N * sizeof(float)), "malloc d_B");
    checkCudaErrors(cudaMalloc((void**)&d_C, M * N * sizeof(float)), "malloc d_C");
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream), "create stream");

    checkCudaErrors(cudaMemcpyAsync(d_A, h_A, M * K * sizeof(float), 
                                    cudaMemcpyHostToDevice, stream), "memcpy h_A->d_A");
    checkCudaErrors(cudaMemcpyAsync(d_B, h_B, K * N * sizeof(float), 
                                    cudaMemcpyHostToDevice, stream), "memcpy h_B->d_B");
    checkCudaErrors(cudaStreamSynchronize(stream), "stream sync");

    auto start = std::chrono::high_resolution_clock::now();
    auto end   = std::chrono::high_resolution_clock::now();
    
    dim3 blockSizeShared(32, 32);
    dim3 gridSizeShared((N + blockSizeShared.x - 1) / blockSizeShared.x,
                        (M + blockSizeShared.y - 1) / blockSizeShared.y);

    start = std::chrono::high_resolution_clock::now();
    matMulSharedKernel<<<gridSizeShared, blockSizeShared, 0, stream>>>(d_A, d_B, d_C, M, K, N);
    checkCudaErrors(cudaStreamSynchronize(stream), "shared kernel sync");
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> shared_ms = end - start;

    checkCudaErrors(cudaMemcpyAsync(h_C, d_C, M * N * sizeof(float), 
    cudaMemcpyDeviceToHost, stream), "memcpy shared result");

    std::cout << "Shared memory:     " << shared_ms.count() << " ms\n";

    cudaStreamDestroy(stream);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
                                    
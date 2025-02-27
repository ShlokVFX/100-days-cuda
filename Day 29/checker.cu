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
    size_t m, size_t k, size_t n)
{
const int TILE_WIDTH = 32;

__shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
__shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

int row = blockDim.y * blockIdx.y + threadIdx.y;
int col = blockDim.x * blockIdx.x + threadIdx.x;

float sum = 0.0f;

int tx = threadIdx.x;
int ty = threadIdx.y;

for (int t = 0; t < (k + TILE_WIDTH - 1) / TILE_WIDTH; t++)
{
int tiledCol = t * TILE_WIDTH + tx;
if (row < m && tiledCol < k)
sharedA[ty][tx] = input_a[row * k + tiledCol];
else
sharedA[ty][tx] = 0.0f;
int tiledRow = t * TILE_WIDTH + ty;
if (tiledRow < k && col < n)
sharedB[ty][tx] = input_b[tiledRow * n + col];
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
if (row < m && col < n)
{
output_c[row * n + col] = sum;
}
}
                             
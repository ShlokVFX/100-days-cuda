#include <stdio.h>
#include <cuda_runtime.h>

__global__ void faultyKernel(int *d_arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bug: No boundary check, leads to out-of-bounds access
    d_arr[idx] = idx * 2;  
}

 

int main() {
    int N = 100;  // Array size
    int *d_arr;

    // Allocate memory on the GPU
    cudaMalloc((void **)&d_arr, N * sizeof(int));

    // Launching more threads than allocated memory
    int blockSize = 32;
    int numBlocks = 5;  // 5 * 32 = 160 threads, but only 100 elements in d_arr
    faultyKernel<<<numBlocks, blockSize>>>(d_arr, N);

    // Error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Synchronize to catch errors
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error after sync: %s\n", cudaGetErrorString(err));
    }

    // Free memory

    // Free memory
    cudaFree(d_arr);
    return 0;
}

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void faultyKernel(int *d_arr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // FIX: Add boundary check to prevent out-of-bounds access
    if (idx < N) {  
        d_arr[idx] = idx * 2;
    }
}

int main() {
    int N = 100;  // Array size
    int *d_arr;

    // Allocate memory on the GPU
    cudaMalloc((void **)&d_arr, N * sizeof(int));

    // Adjust number of threads to match array size
    int blockSize = 32;
    int numBlocks = (N + blockSize - 1) / blockSize; // Ensures we only launch required threads

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
    cudaFree(d_arr);
    return 0;
}

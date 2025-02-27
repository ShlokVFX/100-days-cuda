# CUDA Matrix Multiplication with Shared Memory

This repository contains an optimized CUDA implementation of matrix multiplication using shared memory. The implementation demonstrates efficient GPU parallelization for large matrix multiplications.

## Table of Contents
- [Overview](#overview)
- [Code Breakdown](#code-breakdown)
  - [Error Handling](#error-handling)
  - [CUDA Kernel for Matrix Multiplication](#cuda-kernel-for-matrix-multiplication)
  - [Host Code](#host-code)
  - [Memory Management](#memory-management)
  - [Kernel Launch and Timing](#kernel-launch-and-timing)
  - [Cleanup](#cleanup)
- [Compilation and Execution](#compilation-and-execution)
- [Performance Analysis](#performance-analysis)

## Overview
This CUDA program performs matrix multiplication using a tiled shared memory approach. The goal is to leverage GPU shared memory to improve performance compared to naive implementations. The program supports large matrices and utilizes asynchronous memory transfers with CUDA streams for efficiency.

## Code Breakdown

### Error Handling
```cpp
inline void checkCudaErrors(cudaError_t err, const char* msg = "")
```
A helper function that checks for CUDA errors and reports them. If an error occurs, it prints an error message and exits the program.

### CUDA Kernel for Matrix Multiplication
```cpp
__global__ void matMulSharedKernel(const float* A, const float* B, float* C, int M, int K, int N)
```
This kernel performs matrix multiplication using shared memory tiling:
- Uses `__shared__` memory to store sub-matrices for better performance.
- Computes the matrix product block-wise, iterating over tiled chunks.
- Synchronizes threads to ensure all threads have finished loading shared memory before computing partial results.

### Host Code
#### Matrix Initialization
```cpp
float *h_A, *h_B, *h_C;
checkCudaErrors(cudaMallocHost(&h_A, M * K * sizeof(float)), "malloc pinned h_A");
```
- Allocates pinned memory for host matrices A, B, and C for efficient data transfers.
- Randomly initializes matrices A and B with values between 0 and 1.

### Memory Management
```cpp
float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
checkCudaErrors(cudaMalloc((void**)&d_A, M * K * sizeof(float)), "malloc d_A");
```
- Allocates device memory for matrices A, B, and C.
- Uses `cudaMalloc` to allocate global memory on the GPU.

### Kernel Launch and Timing
```cpp
dim3 blockSizeShared(32, 32);
dim3 gridSizeShared((N + blockSizeShared.x - 1) / blockSizeShared.x, (M + blockSizeShared.y - 1) / blockSizeShared.y);
matMulSharedKernel<<<gridSizeShared, blockSizeShared, 0, stream>>>(d_A, d_B, d_C, M, K, N);
```
- Defines CUDA grid and block dimensions.
- Launches the shared memory kernel asynchronously using CUDA streams.
- Measures execution time using `std::chrono::high_resolution_clock`.

### Cleanup
```cpp
cudaStreamDestroy(stream);
cudaFreeHost(h_A);
cudaFreeHost(h_B);
cudaFreeHost(h_C);
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
```
- Deallocates host and device memory to prevent memory leaks.

## Compilation and Execution
To compile the program, use:
```sh
nvcc matrixMul.cu -o matrixMul
```
To execute:
```sh
./matrixMul
```

## Performance Analysis
The shared memory approach significantly reduces global memory accesses, improving performance. By tiling the matrix multiplication and synchronizing threads, we optimize memory bandwidth and computational efficiency.


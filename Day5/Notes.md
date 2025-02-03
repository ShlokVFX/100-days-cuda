
# Tiled Matrix Multiplication in CUDA

## Overview
This project demonstrates **Tiled Matrix Multiplication** in CUDA using **shared memory** to optimize performance. Tiling improves memory access efficiency by reducing global memory accesses and leveraging **shared memory** for faster computation.

## Prerequisites
Before running the code, ensure you have:
- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit** installed
- **C++ Compiler** that supports CUDA

## How It Works
1. **Allocate Memory**: The program allocates memory for matrices A, B, and C.
2. **Initialize Matrices**: The host initializes input matrices with random values.
3. **Launch Kernel**: The CUDA kernel performs matrix multiplication using tiling.
4. **Synchronize & Print Results**: The program synchronizes computation and prints the result.

## Code Example
```cpp
#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matMulTiled(float *A, float *B, float *C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N / TILE_SIZE); t++) {
        tileA[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE_SIZE + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

int main() {
    int N = 1024; // Matrix size N x N
    size_t bytes = N * N * sizeof(float);
    
    float *A, *B, *C;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);
    
    for (int i = 0; i < N * N; i++) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(N / TILE_SIZE, N / TILE_SIZE);
    
    matMulTiled<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();

    std::cout << "C[0][0] = " << C[0] << ", C[N-1][N-1] = " << C[N*N-1] << std::endl;
    
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    
    return 0;
}
```

## How to Compile & Run
1. **Compile the program**
   ```sh
   nvcc tiled_mat_mul.cu -o tiled_mat_mul
   ```
2. **Run the executable**
   ```sh
   ./tiled_mat_mul
   ```

## Expected Output
```
C[0][0] = 123.456, C[N-1][N-1] = 789.012
```
(Values will vary based on random initialization.)

## Why Use Tiling in Matrix Multiplication?
- **Minimizes global memory accesses** by using **shared memory**
- **Reduces memory latency** and improves parallel efficiency
- **Essential for optimizing large matrix computations**

## License
This project is licensed under the MIT License.


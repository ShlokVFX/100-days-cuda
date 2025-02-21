# Unified Memory Vector Addition in CUDA

## Overview
This project demonstrates **Unified Memory** in CUDA by performing **vector addition**. Unified Memory simplifies memory management by allowing the CPU and GPU to share the same memory space.

## Prerequisites
Before running the code, ensure you have:
- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit** installed
- **C++ Compiler** that supports CUDA

## How It Works
1. **Allocate Unified Memory**: The program uses `cudaMallocManaged()` to allocate memory accessible by both the CPU and GPU.
2. **Initialize Vectors**: The host initializes the input vectors.
3. **Launch Kernel**: A CUDA kernel performs element-wise vector addition in parallel.
4. **Synchronize & Print Results**: The program synchronizes with `cudaDeviceSynchronize()` and prints the result.

## Code Example
```cpp
#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(int *a, int *b, int *c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int N = 1 << 20; // 1M elements
    int *a, *b, *c;
    
    // Allocate Unified Memory
    cudaMallocManaged(&a, N * sizeof(int));
    cudaMallocManaged(&b, N * sizeof(int));
    cudaMallocManaged(&c, N * sizeof(int));

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    vectorAdd<<<numBlocks, blockSize>>>(a, b, c, N);

    // Synchronize
    cudaDeviceSynchronize();

    // Print sample result
    std::cout << "c[0] = " << c[0] << ", c[N-1] = " << c[N-1] << std::endl;

    // Free memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
```

## How to Compile & Run
1. **Compile the program**
   ```sh
   nvcc vector_add_unified_memory.cu -o vector_add
   ```
2. **Run the executable**
   ```sh
   ./vector_add
   ```

## Expected Output
```
c[0] = 0, c[N-1] = 3145722
```

## Why Use Unified Memory?
- **No explicit `cudaMemcpy` calls** between host and device
- **Easier memory management** for beginners
- **Ideal for prototyping CUDA applications**

## License
This project is licensed under the MIT License.


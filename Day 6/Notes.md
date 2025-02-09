# Optimized CUDA Matrix Transpose

## Description
This repository contains an optimized CUDA implementation of matrix transposition using shared memory. The code improves performance by:

- Using **shared memory** with padding to avoid bank conflicts.
- Ensuring **coalesced global memory access** for efficient data transfer.
- Employing **proper thread synchronization** using `__syncthreads()`.
- Using **CUDA event timing** for precise performance measurement.

## Ensuring Coalesced Global Memory Access
Coalesced memory access refers to a pattern where multiple threads in a warp access consecutive memory locations, leading to efficient memory transactions. In CUDA, global memory is accessed in 32-, 64-, or 128-byte memory transactions, so ensuring that threads access memory in a sequential, aligned manner minimizes memory latency.

### How This Code Achieves Coalesced Access:
1. **Row-wise Read from Global Memory**:
   - Each thread reads an element from a row in a **sequential** manner, ensuring minimal memory transactions.
2. **Shared Memory Usage**:
   - The data is stored in shared memory with padding to **avoid bank conflicts**, allowing efficient memory reuse.
3. **Column-wise Write to Global Memory**:
   - The transposed data is written back in a way that ensures each warp accesses contiguous memory, optimizing memory throughput.

## Requirements
To compile and run the code, ensure you have:
- **CUDA Toolkit** installed
- A **NVIDIA GPU** that supports CUDA
- A compatible **C++ compiler (e.g., nvcc, GCC)**

## Compilation & Execution

1. **Compile the code using NVCC:**
    ```sh
    nvcc -o transpose transpose.cu
    ```
2. **Run the executable:**
    ```sh
    ./transpose
    ```

## Expected Output
Upon execution, the program prints the execution time of the CUDA kernel:
```sh
Optimized Kernel Execution time: X.XX ms
```
Where `X.XX ms` is the measured execution time of the transpose kernel.

## File Structure
```
├── transpose.cu  # Optimized CUDA matrix transpose implementation
├── README.md     # Documentation
```

## Code
```cpp
#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

// Optimized matrix transpose kernel using shared memory
__global__ void transposeKernel(float* input, float* output, int width, int height) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1]; // Padding to avoid bank conflicts

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }

    __syncthreads();

    int transposedX = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    int transposedY = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    if (transposedX < height && transposedY < width) {
        output[transposedY * height + transposedX] = tile[threadIdx.x][threadIdx.y];
    }
}

void transpose(float* h_input, float* h_output, int width, int height, float& time) {
    float *d_input, *d_output;

    cudaMalloc((void**)&d_input, width * height * sizeof(float));
    cudaMalloc((void**)&d_output, width * height * sizeof(float));

    cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    transposeKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize(); // Ensure kernel execution completes before timing

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaMemcpy(h_output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int width = 1024;
    int height = 1024;

    float* h_input = (float*)malloc(width * height * sizeof(float));
    float* h_output = (float*)malloc(width * height * sizeof(float));

    for (int i = 0; i < width * height; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    float time = 0.0f;
    transpose(h_input, h_output, width, height, time);

    std::cout << "Optimized Kernel Execution time: " << time << "ms" << std::endl;

    free(h_input);
    free(h_output);

    return 0;
}
```

## Performance Optimizations
- **Shared Memory Usage**: Reduces redundant global memory accesses.
- **Memory Coalescing**: Enhances global memory throughput.
- **Loop Unrolling (if applicable)**: Reduces overhead and improves speed.

## License
This project is open-source and available under the **MIT License**.

## Author
Developed as part of the **100 Days of Learning CUDA Kernels** challenge.


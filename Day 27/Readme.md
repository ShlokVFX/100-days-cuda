# Matrix Multiplication in CUDA

This project demonstrates matrix multiplication using CUDA. CUDA (Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) model created by NVIDIA.

## Prerequisites

- NVIDIA GPU
- CUDA Toolkit
- C++ Compiler

## Getting Started

1. **Clone the repository:**
    ```sh
    git clone <repository_url>
    cd 100-days-cuda/Day-27
    ```

2. **Build the project:**
    ```sh
    nvcc matrixmul.cu -o matrixmul
    ```

3. **Run the executable:**
    ```sh
    ./matrixmul
    ```

## Code Explanation

### Kernel Function

The core of the matrix multiplication is the kernel function, which runs on the GPU:

```cpp
__global__ void matrixMulKernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (row < N && col < N) {
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

### Host Code

The host code sets up the matrices and calls the kernel:

```cpp
int main() {
    int N = 1024; // Matrix size
    size_t size = N * N * sizeof(float);

    // Allocate memory on host
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize matrices h_A and h_B
    // ...

    // Allocate memory on device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```

## Conclusion

This example demonstrates the basics of matrix multiplication using CUDA. By leveraging the parallel processing power of GPUs, significant performance improvements can be achieved for large matrix operations.

For more details, refer to the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).

# Day 35: Understanding ReLU and Implementing it in CUDA

## Introduction to ReLU

ReLU stands for Rectified Linear Unit. It is a popular activation function used in neural networks, especially in deep learning models. The ReLU function is defined as:

\[ \text{ReLU}(x) = \max(0, x) \]

This means that if the input is positive, it returns the input itself; otherwise, it returns zero. ReLU helps in introducing non-linearity to the model while being computationally efficient.

## Benefits of ReLU

- **Simplicity**: Easy to implement and compute.
- **Non-linearity**: Introduces non-linearity, which helps in learning complex patterns.
- **Sparse Activation**: Only a few neurons are activated at a time, making the network efficient.

## CUDA Implementation of ReLU

Below is a sample CUDA code to implement the ReLU activation function:

```cuda
#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel to apply ReLU activation function
__global__ void relu(float* d_input, float* d_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_output[idx] = fmaxf(0.0f, d_input[idx]);
    }
}

int main() {
    const int size = 10;
    float h_input[size] = {-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0};
    float h_output[size];

    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    relu<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size);

    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "ReLU Output: ";
    for (int i = 0; i < size; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
```

## Explanation

1. **Kernel Function**: The `relu` kernel function applies the ReLU operation to each element of the input array.
2. **Memory Allocation**: Memory is allocated on the GPU for input and output arrays.
3. **Data Transfer**: Input data is copied from the host (CPU) to the device (GPU).
4. **Kernel Launch**: The kernel is launched with an appropriate number of blocks and threads.
5. **Result Transfer**: The result is copied back from the device to the host.
6. **Cleanup**: GPU memory is freed.

This code demonstrates a simple and efficient way to apply the ReLU activation function using CUDA.

#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define BLOCK_SIZE 8  // This remains unchanged to support larger matrices

// CUDA Kernel for Matrix Transposition using Shared Memory
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

// Function to print matrices
void printMatrix(const char* name, float* matrix, int width, int height) {
    std::cout << name << ":\n";
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << matrix[i * width + j] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// Host function to perform matrix transposition
void transpose(float* h_input, float* h_output, int width, int height, float& time) {
    float* d_input, * d_output;

    cudaMalloc((void**)&d_input, width * height * sizeof(float));
    cudaMalloc((void**)&d_output, width * height * sizeof(float));

    cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    transposeKernel <<< gridDim, blockDim >>> (d_input, d_output, width, height);
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
    int width = 3;
    int height = 3;

    float h_input[9] = { 1, 2, 3, 
                         4, 5, 6, 
                         7, 8, 9 };

    float h_output[9] = { 0 };

    float time = 0.0f;
    transpose(h_input, h_output, width, height, time);

    // Print matrices
    printMatrix("Input Matrix", h_input, width, height);
    printMatrix("Transposed Matrix", h_output, height, width);

    std::cout << "Optimized Kernel Execution time: " << time << " ms\n";

    return 0;
}

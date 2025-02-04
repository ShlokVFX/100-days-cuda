#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

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

    transposeKernel << <gridDim, blockDim >> > (d_input, d_output, width, height);
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

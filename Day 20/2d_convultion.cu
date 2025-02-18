#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void conv2D(float *input, float *output, float *mask, int maskSize, int inputSize) {
    __shared__ float sharedInput[TILE_WIDTH + 2][TILE_WIDTH + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    int maskRadius = maskSize / 2;

    if (row < inputSize && col < inputSize) {
        sharedInput[ty + maskRadius][tx + maskRadius] = input[row * inputSize + col];
        if (ty < maskRadius) {
            if (row >= maskRadius) {
                sharedInput[ty][tx + maskRadius] = input[(row - maskRadius) * inputSize + col];
            } else {
                sharedInput[ty][tx + maskRadius] = 0.0f;
            }
            if (row + TILE_WIDTH < inputSize) {
                sharedInput[ty + TILE_WIDTH + maskRadius][tx + maskRadius] = input[(row + TILE_WIDTH) * inputSize + col];
            } else {
                sharedInput[ty + TILE_WIDTH + maskRadius][tx + maskRadius] = 0.0f;
            }
        }
        if (tx < maskRadius) {
            if (col >= maskRadius) {
                sharedInput[ty + maskRadius][tx] = input[row * inputSize + col - maskRadius];
            } else {
                sharedInput[ty + maskRadius][tx] = 0.0f;
            }
            if (col + TILE_WIDTH < inputSize) {
                sharedInput[ty + maskRadius][tx + TILE_WIDTH + maskRadius] = input[row * inputSize + col + TILE_WIDTH];
            } else {
                sharedInput[ty + maskRadius][tx + TILE_WIDTH + maskRadius] = 0.0f;
            }
        }
        if (ty < maskRadius && tx < maskRadius) {
            if (row >= maskRadius && col >= maskRadius) {
                sharedInput[ty][tx] = input[(row - maskRadius) * inputSize + col - maskRadius];
            } else {
                sharedInput[ty][tx] = 0.0f;
            }
            if (row + TILE_WIDTH < inputSize && col + TILE_WIDTH < inputSize) {
                sharedInput[ty + TILE_WIDTH + maskRadius][tx + TILE_WIDTH + maskRadius] = input[(row + TILE_WIDTH) * inputSize + col + TILE_WIDTH];
            } else {
                sharedInput[ty + TILE_WIDTH + maskRadius][tx + TILE_WIDTH + maskRadius] = 0.0f;
            }
        }
    }

    __syncthreads();

    if (row < inputSize && col < inputSize) {
        float result = 0.0f;
        for (int i = 0; i < maskSize; i++) {
            for (int j = 0; j < maskSize; j++) {
                result += sharedInput[ty + i][tx + j] * mask[i * maskSize + j];
            }
        }
        output[row * inputSize + col] = result;
    }
}

void measureGflops(float *input, float *output, float *mask, int maskSize, int inputSize) {
    float *d_input, *d_output, *d_mask;
    cudaMalloc(&d_input, inputSize * inputSize * sizeof(float));
    cudaMalloc(&d_output, inputSize * inputSize * sizeof(float));
    cudaMalloc(&d_mask, maskSize * maskSize * sizeof(float));

    cudaMemcpy(d_input, input, inputSize * inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, maskSize * maskSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((inputSize + TILE_WIDTH - 1) / TILE_WIDTH, (inputSize + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    conv2D<<<dimGrid, dimBlock>>>(d_input, d_output, d_mask, maskSize, inputSize);
    cudaEventRecord(stop);

    cudaMemcpy(output, d_output, inputSize * inputSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Time: %f ms\n", milliseconds);
    printf("GFLOPS: %f\n", 2.0 * inputSize * inputSize * maskSize * maskSize / milliseconds / 1e6);
    printf("Output: %f\n", output[0]);
    std::cout << "inputSize: " << inputSize << std::endl;
    std::cout << "maskSize: " << maskSize << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);
}

int main() {
    int inputSize = 1024;
    int maskSize = 5;

    float *input = new float[inputSize * inputSize];
    float *output = new float[inputSize * inputSize];
    float *mask = new float[maskSize * maskSize];

    for (int i = 0; i < inputSize * inputSize; i++) {
        input[i] = static_cast<float>(i);
    }

    for (int i = 0; i < maskSize * maskSize; i++) {
        mask[i] = static_cast<float>(i);
    }

    measureGflops(input, output, mask, maskSize, inputSize);

    delete[] input;
    delete[] output;
    delete[] mask;

    return 0;
}
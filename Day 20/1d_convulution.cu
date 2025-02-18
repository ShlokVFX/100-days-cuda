#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void conv1D(float *input, float *output, float *mask, int inputSize, int maskSize) {
    __shared__ float sharedInput[TILE_WIDTH + 2];

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int index = bx * TILE_WIDTH + tx;

    int maskRadius = maskSize / 2;

    if (index < inputSize) {
        sharedInput[tx + maskRadius] = input[index];
        if (tx < maskRadius) {
            if (index >= maskRadius) {
                sharedInput[tx] = input[index - maskRadius];
            } else {
                sharedInput[tx] = 0.0f;
            }
            if (index + TILE_WIDTH < inputSize) {
                sharedInput[tx + TILE_WIDTH + maskRadius] = input[index + TILE_WIDTH];
            } else {
                sharedInput[tx + TILE_WIDTH + maskRadius] = 0.0f;
            }
        }
    }

    __syncthreads();

    if (index < inputSize) {
        float result = 0.0f;
        for (int i = 0; i < maskSize; i++) {
            result += sharedInput[tx + i] * mask[i];
        }
        output[index] = result;
    }
}

void measureGflops(float *input, float *output, float *mask, int inputSize, int maskSize) {
    float *d_input, *d_output, *d_mask;
    cudaMalloc(&d_input, inputSize * sizeof(float));
    cudaMalloc(&d_output, inputSize * sizeof(float));
    cudaMalloc(&d_mask, maskSize * sizeof(float));

    cudaMemcpy(d_input, input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, maskSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, 1, 1);
    dim3 dimGrid((inputSize + TILE_WIDTH - 1) / TILE_WIDTH, 1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    conv1D<<<dimGrid, dimBlock>>>(d_input, d_output, d_mask, inputSize, maskSize);
    cudaEventRecord(stop);

    cudaMemcpy(output, d_output, inputSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    
    printf("Time: %f ms\n", milliseconds);
    printf("GFLOPS: %f\n", 2.0 * inputSize * maskSize / (milliseconds * 1e6));
    printf("Output: %f\n", output[0]);
    printf("inputSize: %d\n", inputSize);
    printf("maskSize: %d\n", maskSize);
   // std::cout << "GFLOPS: " << gflops << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);
}

int main() {
    int inputSize = 1024;
    int maskSize = 5;

    float *input = new float[inputSize];
    float *output = new float[inputSize];
    float *mask = new float[maskSize];

    for (int i = 0; i < inputSize; i++) {
        input[i] = static_cast<float>(i);
    }

    for (int i = 0; i < maskSize; i++) {
        mask[i] = static_cast<float>(i);
    }

    measureGflops(input, output, mask, inputSize, maskSize);

    delete[] input;
    delete[] output;
    delete[] mask;

    return 0;
}

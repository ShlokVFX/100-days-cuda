#include<iostream>
#include<cuda_runtime.h>

// matrix transpose kernel
__global__ void transposeKernel(float* input, float* output, int width, int height) {

    // use of shared memory
    __shared__ float tile[32][32 + 1];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }

    // synchronize the threads
    __syncthreads();

    // calculate the transposed matrix
    int transposedX = blockIdx.y * blockDim.y + threadIdx.y;
    int transposedY = blockIdx.x * blockDim.x + threadIdx.x;

    if (transposedX < height && transposedY < width) {
        output[transposedY * height + transposedX] = tile[threadIdx.x][threadIdx.y];
    }
}

// host func to launch the kernel
void transpose(float* h_input, float* h_output, int width, int height, float& time) {
    float* d_input, * d_output;

    cudaMalloc((void**)&d_input, width * height * sizeof(float));
    cudaMalloc((void**)&d_output, width * height * sizeof(float));

    cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    transposeKernel << <gridDim, blockDim >> > (d_input, d_output, width, height);

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

    std::cout << "Kernel Execution time: " << time << "ms" << std::endl;

    free(h_input);
    free(h_output);

    return 0;
}
// Kernel Execution time: 0.75s
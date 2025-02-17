#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256 // Define block size as a constant
#define CUDA_CHECK(call)                                                          \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                  \
        }                                                                         \
    } while (0)

__global__ void sumReduction(int *input, int *output, int size) {
    extern __shared__ int sharedData[];

    unsigned int tid = threadIdx.x;
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[tid] = (index < size) ? input[index] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

int main() {
    const int size = 1024;
    const int bytes = size * sizeof(int);

    int h_input[size];
    for (int i = 0; i < size; i++) {
        h_input[i] = 1; // Initialize array with 1s
    }

    int *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));

    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_CHECK(cudaMalloc(&d_output, blocks * sizeof(int))); // Correct allocation size

    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    sumReduction<<<blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_input, d_output, size);

    int h_output[blocks];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, blocks * sizeof(int), cudaMemcpyDeviceToHost));

    int totalSum = 0;
    for (int i = 0; i < blocks; i++) {
        totalSum += h_output[i];
    }

    printf("Sum: %d\n", totalSum);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}

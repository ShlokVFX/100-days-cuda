#include <cuda_runtime.h>

__global__ void relu_kernel(float* input, float* output, size_t n, size_t m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * m;

    if (idx < total) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

extern "C" void solution(float* input, float* output, size_t n, size_t m) {
    int total = n * m;
    int threadsPerBlock = 688;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, n, m);
    cudaDeviceSynchronize();
}
#include <cuda_runtime.h>

__global__ void matmul_symmetric_kernel(const float* A, const float* B, float* C, size_t N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0.0f;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

extern "C" void solution(float* input_a, float* input_b, float* output_c, size_t N) {
    dim3 blockSize(16, 16); 
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);

    matmul_symmetric_kernel<<<gridSize, blockSize>>>(input_a, input_b, output_c, N);

    cudaDeviceSynchronize();
}

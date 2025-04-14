#include <cuda_runtime.h>

__global__ void batched_tensor_matmul_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* C,                    
    size_t N, size_t M, size_t K, size_t L)
{
    int n = blockIdx.z;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int l = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N && m < M && l < L) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[n * M * K + m * K + k] * B[k * L + l];
        }
        C[n * M * L + m * L + l] = sum;
    }
}

extern "C" void solution(float* input_a, float* input_b, float* output_c,
                         size_t N, size_t M, size_t K, size_t L)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((L + 15) / 16, (M + 15) / 16, N);

    batched_tensor_matmul_kernel<<<gridSize, blockSize>>>(input_a, input_b, output_c, N, M, K, L);

    cudaDeviceSynchronize();
}

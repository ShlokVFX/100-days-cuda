#include <cuda_runtime.h>

__global__ void tensor4d_matmul_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B,
    float* C,         
    size_t b, size_t i, size_t j, size_t l, size_t k)
{
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < b && row < i && col < j) {
        for (int out_k = 0; out_k < k; ++out_k) {
            float sum = 0.0f;
            for (int inner = 0; inner < l; ++inner) {
                size_t a_idx = ((batch * i + row) * j + col) * l + inner;
                size_t b_idx = inner * k + out_k;                   
                sum += A[a_idx] * B[b_idx];
            }
            size_t c_idx = ((batch * i + row) * j + col) * k + out_k;
            C[c_idx] = sum;
        }
    }
}

extern "C" void solution(float* input_a, float* input_b, float* output_c,
                         size_t b, size_t i, size_t j, size_t l, size_t k)
{
    dim3 blockSize(16, 16); i)
    dim3 gridSize((j + 15) / 16, (i + 15) / 16, b);

    tensor4d_matmul_kernel<<<gridSize, blockSize>>>(input_a, input_b, output_c, b, i, j, l, k);

    cudaDeviceSynchronize();
}

#include <cuda_runtime.h>

__global__ void vector_add(const float* __restrict__ d_input1,
                           const float* __restrict__ d_input2,
                           float* __restrict__ d_output,
                           size_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_output[idx] = d_input1[idx] + d_input2[idx];
    }
}

__global__ void vector_add_vec2(const float* __restrict__ d_input1,
                                const float* __restrict__ d_input2,
                                float* __restrict__ d_output,
                                size_t n_vec2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec2) {
        float2 a = reinterpret_cast<const float2*>(d_input1)[idx];
        float2 b = reinterpret_cast<const float2*>(d_input2)[idx];
        float2 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        reinterpret_cast<float2*>(d_output)[idx] = c;
    }
}

extern "C" void solution(float* d_input1, float* d_input2, float* d_output, size_t n) {
    size_t n_vec2 = n / 2;
    size_t remainder = n % 2;

    int threadsPerBlock = 512;
    int blocksPerGrid = (n_vec2 + threadsPerBlock - 1) / threadsPerBlock; // Fixed grid size calculation

    if (n_vec2 > 0) {
        vector_add_vec2<<<blocksPerGrid, threadsPerBlock>>>(d_input1, d_input2, d_output, n_vec2);
    }
    if (remainder > 0) {
        size_t offset = n - remainder;
        blocksPerGrid = (remainder + threadsPerBlock - 1) / threadsPerBlock;
        vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_input1 + offset, d_input2 + offset, d_output + offset, remainder);
    }
}

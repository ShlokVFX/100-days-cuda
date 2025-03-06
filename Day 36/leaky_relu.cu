#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float4* __restrict__ input, float4* __restrict__ output, size_t n, size_t m, float alpha) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (row < n && col < m) {
        int idx = row * (m / 4) + (col / 4);
        float4 val = __ldg(&input[idx]);
        val.x = val.x * (val.x >= 0.0f) + alpha * val.x * (val.x < 0.0f);
        val.y = val.y * (val.y >= 0.0f) + alpha * val.y * (val.y < 0.0f);
        val.z = val.z * (val.z >= 0.0f) + alpha * val.z * (val.z < 0.0f);
        val.w = val.w * (val.w >= 0.0f) + alpha * val.w * (val.w < 0.0f);
        output[idx] = val;
    }
}

extern "C" void solution(float* input, float* output, size_t n, size_t m, float alpha) {
    float4* input4 = reinterpret_cast<float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);
    size_t m4 = m / 4;
    dim3 blockSize(512, 1);  
    dim3 gridSize((m4 + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    leaky_relu_kernel<<<gridSize, blockSize>>>(input4, output4, n, m, alpha);
    cudaDeviceSynchronize();
}

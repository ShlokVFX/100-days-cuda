#include <cuda_runtime.h>

__global__ void vector_add_fused(const float* __restrict__ d_input1,
                                 const float* __restrict__ d_input2,
                                 float* __restrict__ d_output,
                                 size_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_vec2 = n / 2;

    if (idx < n_vec2) {
        float2 a = reinterpret_cast<const float2*>(d_input1)[idx];
        float2 b = reinterpret_cast<const float2*>(d_input2)[idx];
        float2 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        reinterpret_cast<float2*>(d_output)[idx] = c;
    }
    
    // Handle the remaining elements (if n is odd)
    if (idx == n_vec2 && (n % 2) != 0) {
        d_output[n - 1] = d_input1[n - 1] + d_input2[n - 1];
    }
}

extern "C" void solution(float* d_input1, float* d_input2, float* d_output, size_t n) {
    
    int threadsPerBlock = 512;
    int blocksPerGrid = ((n / 2) + threadsPerBlock - 1) / threadsPerBlock;
    vector_add_fused<<<blocksPerGrid, threadsPerBlock>>>(d_input1, d_input2, d_output, n);
}


//this is part of cmake file.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>  // ✅ Fix for uintptr_t
#include <cstdio>   // ✅ Fix for printf

// ✅ Use CUDA built-in function instead of inline PTX
__device__ inline half2 my_hadd2(half2 a, half2 b) {
    return __hadd2(a, b);  // ✅ Built-in FP16 vectorized addition
}

// ✅ Optimized vector addition kernel using FP16 (half2)
__global__ void vector_add_half2(const half2* __restrict__ d_input1,
                                 const half2* __restrict__ d_input2,
                                 half2* __restrict__ d_output,
                                 size_t n_half2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_half2) {
        half2 a = __ldg(&d_input1[idx]);  // ✅ Cached read
        half2 b = __ldg(&d_input2[idx]);

        half2 c = my_hadd2(a, b);
        d_output[idx] = c;
    }
}

// ✅ Optimized `vector_add_vec2` kernel using `float2`
__global__ void vector_add_vec2(const float2* __restrict__ d_input1,
                                const float2* __restrict__ d_input2,
                                float2* __restrict__ d_output,
                                size_t n_vec2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec2) {
        float2 a = __ldg(&d_input1[idx]);  // ✅ Cached read
        float2 b = __ldg(&d_input2[idx]);

        float2 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        d_output[idx] = c;  // ✅ Store addition result safely

        if (idx < n_vec2 - 1) {  // ✅ Prevent out-of-bounds write
            float2 d;
            d.x = a.x * b.x;
            d.y = a.y * b.y;
            d_output[idx + 1] = d;  // ✅ Only store multiplication if there's space
        }
    }
}

// ✅ Optimized Solution Function
extern "C" void solution(float* d_input1, float* d_input2, float* d_output, size_t n) {
    size_t n_vec2 = n / 2;
    size_t remainder = n % 2;

    int threadsPerBlock = (n < 10'000'00) ? 256 : 512;
    int blocksPerGrid = (n_vec2 + threadsPerBlock - 1) / threadsPerBlock;

    if (n_vec2 > 0) {
        vector_add_vec2<<<blocksPerGrid, threadsPerBlock>>>(
            reinterpret_cast<const float2*>(d_input1),  
            reinterpret_cast<const float2*>(d_input2),
            reinterpret_cast<float2*>(d_output),
            n_vec2);
    }

    if (remainder > 0) {
        size_t offset = n - remainder;
        blocksPerGrid = (remainder + threadsPerBlock - 1) / threadsPerBlock;

        // 🚨 Ensure `half2` alignment (4 bytes)
        if ((reinterpret_cast<std::uintptr_t>(d_input1 + offset) % 4) == 0) {
            vector_add_half2<<<blocksPerGrid, threadsPerBlock>>>(
                reinterpret_cast<const half2*>(d_input1 + offset),
                reinterpret_cast<const half2*>(d_input2 + offset),
                reinterpret_cast<half2*>(d_output + offset),
                remainder / 2);
        } else {
            printf("🚨 Error: Memory misaligned for half2 access!\n");
        }
    }

    // ✅ Debugging: Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Error: %s\n", cudaGetErrorString(err));
    }
}

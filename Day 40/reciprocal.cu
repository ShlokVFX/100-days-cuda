#include <iostream>
#include <cuda_runtime.h>

__global__ void fastMathExample(float *d_out, float *d_in) {
    int tid = threadIdx.x;
    float val = d_in[tid];
    float result;

    // Compute reciprocal using PTX
    asm volatile("rcp.approx.f32 %0, %1;" : "=f"(result) : "f"(val));
    
    d_out[tid] = result;
}

int main() {
    const int N = 8;
    float h_in[N] = {1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f};
    float h_out[N];
    float *d_in, *d_out;

    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    fastMathExample<<<1, N>>>(d_out, d_in);

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Reciprocal values: ";
    for (int i = 0; i < N; i++) std::cout << h_out[i] << " ";
    std::cout << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}

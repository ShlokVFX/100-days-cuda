#include <iostream>
#include <cuda_runtime.h>

__global__ void membarExample(int *d_out, int *d_in) {
    int tid = threadIdx.x;
    int val = d_in[tid];

    // Memory fence for global memory
    asm volatile("membar.gl;");
    
    d_out[tid] = val;
}

int main() {
    const int N = 32;
    int h_in[N], h_out[N];
    int *d_in, *d_out;

    for (int i = 0; i < N; i++) h_in[i] = i;

    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));

    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    membarExample<<<1, N>>>(d_out, d_in);

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Memory fence applied, values: ";
    for (int i = 0; i < N; i++) std::cout << h_out[i] << " ";
    std::cout << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}

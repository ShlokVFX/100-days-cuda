#include <iostream>
#include <cuda_runtime.h>

__global__ void bitwiseExample(int *d_out, int *d_in) {
    int tid = threadIdx.x;
    int val = d_in[tid];
    int result;

    // Count the number of set bits
    asm volatile("popc.b32 %0, %1;" : "=r"(result) : "r"(val));
    
    d_out[tid] = result;
}

int main() {
    const int N = 8;
    int h_in[N] = {0b0001, 0b0011, 0b0111, 0b1111, 0b1000, 0b1100, 0b1010, 0b1110};
    int h_out[N];
    int *d_in, *d_out;

    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));

    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    bitwiseExample<<<1, N>>>(d_out, d_in);

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Number of set bits: ";
    for (int i = 0; i < N; i++) std::cout << h_out[i] << " ";
    std::cout << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}

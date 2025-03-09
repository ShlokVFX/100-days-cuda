#include <iostream>
#include <cuda_runtime.h>

__global__ void inlinePTXKernel(int *d_out, int *d_in, int addVal) {
    int tid = threadIdx.x;
    int val;

    // Load a value from global memory using PTX
    asm volatile("ld.global.s32 %0, [%1];" : "=r"(val) : "l"(d_in + tid));

    // Add a constant using PTX
    asm volatile("add.s32 %0, %0, %1;" : "+r"(val) : "r"(addVal));

    // Store the result back to global memory using PTX
    asm volatile("st.global.s32 [%0], %1;" :: "l"(d_out + tid), "r"(val));
}

int main() {
    const int N = 32;
    int h_in[N], h_out[N];
    int *d_in, *d_out;
    int addVal = 10;

    for (int i = 0; i < N; i++) h_in[i] = i;

    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    inlinePTXKernel<<<1, N>>>(d_out, d_in, addVal);

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Results: ";
    for (int i = 0; i < N; i++)
        std::cout << h_out[i] << " ";
    std::cout << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}

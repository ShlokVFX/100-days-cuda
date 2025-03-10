#include <iostream>
#include <cuda_runtime.h>

__global__ void inlinePTXExample(int *d_out, int *d_in, int addVal) {
    int tid = threadIdx.x;
    int val;
    int result;

   asm volatile("ld.global.s32 %0,[%1];" : "=r"(val) : "l"(d_in + tid)); //Load value from global

   asm volatile("add.s32 %0 , %0, %1;" : "+r"(val) : "r"(addVal)); // Add constant

   asm volatile("st.global.s32 [%0], %1;" :: "l"(d_out + tid), "r"(val)); // Store value to global

   asm volatile("shfl.sync.bfly %0, %1 , 0, 0x1f;" : "=r"(result) : "r"(val)); // Shuffle

   d_out[tid] = result;


}

int main() {
    const int N = 32;
    int h_in[N], h_out[N];
    int *d_in, *d_out;
    int addVal =10;

    for (int i = 0; i < N; i++) h_in[i] = i * 2;

    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));

    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    inlinePTXExample<<<1, N>>>(d_out, d_in, addVal);

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        std::cout << h_out[i] << " ";

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

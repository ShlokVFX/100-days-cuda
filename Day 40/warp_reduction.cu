#include <iostream>
#include <cuda_runtime.h>

__global__ void reductionExample(int *d_out, int *d_in) {
    int val = d_in[threadIdx.x];

    // Use the built-in shuffle function
    for (int offset = 16; offset > 0; offset /= 2) {
        int temp = __shfl_down_sync(0xffffffff, val, offset);
        val += temp;
    }

    if (threadIdx.x == 0)
        d_out[0] = val;
}

int main() {
    const int N = 32;
    int h_in[N], h_out;
    int *d_in, *d_out;

    // Initialize with 1s
    for (int i = 0; i < N; i++) h_in[i] = 1 * 2;

    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, sizeof(int));

    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    reductionExample<<<1, N>>>(d_out, d_in);

    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Warp Reduction Result: " << h_out << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}

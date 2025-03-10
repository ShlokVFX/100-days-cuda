#include <iostream>
#include <cuda_runtime.h>

__global__ void shuffleExample(int *d_out, int *d_in) {
    int tid = threadIdx.x;
    int val = d_in[tid];
    // Broadcast the value from lane 0 to all threads in the warp.
    int result = __shfl_sync(0xFFFFFFFF, val, 1);
    d_out[tid] = result;
}

int main() {
    const int N = 32;
    int h_in[N], h_out[N];
    int *d_in, *d_out;
    
    // Initialize input data.
    for (int i = 0; i < N; i++) {
        h_in[i] = i + 1;
    }

    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel with one warp (32 threads).
    shuffleExample<<<1, N>>>(d_out, d_in);
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Broadcasted value from thread 1: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

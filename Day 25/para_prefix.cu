//Parallel Prefix Sum (Scan)

//A parallel prefix sum (scan) can be implemented using Hillis-Steele (work-efficient)
//or Blelloch's (less synchronization) algorithm. Below is an implementation of the Blelloch scan using shared memory for efficiency.

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

__global__ void prefixSumExclusive(int *d_in, int *d_out, int N) {
    __shared__ int temp[BLOCK_SIZE * 2];

    int tid = threadIdx.x;
    int offset = 1;

    int ai = tid;
    int bi = tid + (N / 2);

    temp[ai] = (ai < N) ? d_in[ai] : 0;
    temp[bi] = (bi < N) ? d_in[bi] : 0;

    for (int d = N >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (tid == 0) temp[N - 1] = 0;

    for (int d = 1; d < N; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    __syncthreads();
    if (ai < N) d_out[ai] = temp[ai];
    if (bi < N) d_out[bi] = temp[bi];
}

int main() {
    int N = 1024;
    int h_in[N], h_out[N];

    for (int i = 0; i < N; i++) h_in[i] = i + 1;

    int *d_in, *d_out;
    cudaMalloc((void**)&d_in, N * sizeof(int));
    cudaMalloc((void**)&d_out, N * sizeof(int));

    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    prefixSumExclusive<<<1, BLOCK_SIZE>>>(d_in, d_out, N);

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Exclusive Prefix Sum:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

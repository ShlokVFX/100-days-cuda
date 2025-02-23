#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

__global__ void prefixSumExclusive(int *d_in, int *d_out, int N) {
    __shared__ int temp[BLOCK_SIZE * 2];

    int tid = threadIdx.x;
    int offset = 1;

    int ai = tid;
    int bi = tid + (N / 2);

    // Load data into shared memory
    temp[ai] = (ai < N) ? d_in[ai] : 0;
    temp[bi] = (bi < N) ? d_in[bi] : 0;

    // Reduction phase (upsweep)
    for (int d = N >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int indexA = offset * (2 * tid + 1) - 1;
            int indexB = offset * (2 * tid + 2) - 1;
            temp[indexB] += temp[indexA];
        }
        offset *= 2;
    }

    // Set last element to 0 for exclusive scan
    if (tid == 0) temp[N - 1] = 0;

    // Downsweep phase
    for (int d = 1; d < N; d *= 2) {
        offset /= 2;
        __syncthreads();
        if (tid < d) {
            int indexA = offset * (2 * tid + 1) - 1;
            int indexB = offset * (2 * tid + 2) - 1;
            int t = temp[indexA];
            temp[indexA] = temp[indexB];
            temp[indexB] += t;
        }
    }
    __syncthreads();

    // Write results back to global memory
    if (ai < N) d_out[ai] = temp[ai];
    if (bi < N) d_out[bi] = temp[bi];
}

int main() {
    int N = 1024;
    int h_in[N], h_out[N];

    // Initialize input array with values 1 to 1024
    for (int i = 0; i < N; i++) {
        h_in[i] = i + 1;
    }

    int *d_in, *d_out;
    cudaMalloc((void**)&d_in, N * sizeof(int));
    cudaMalloc((void**)&d_out, N * sizeof(int));

    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // Setup CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record the start event
    cudaEventRecord(start, 0);
    
    // Launch kernel (1 block, BLOCK_SIZE threads)
    prefixSumExclusive<<<1, BLOCK_SIZE>>>(d_in, d_out, N);
    
    // Record the stop event and wait for the kernel to finish
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Calculate total additions: approximately 2*(N - 1)
    long totalOps = 2 * (N - 1);
    // Convert elapsed time to seconds
    float seconds = milliseconds / 1000.0f;
    // Calculate GFLOPS (Giga operations per second)
    float gflops = totalOps / (seconds * 1e9);
    
    // Print scan result
    printf("Scan found (exclusive prefix sum):\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    // Print benchmark results
    printf("Kernel execution time: %f ms\n", milliseconds);
    printf("Performance: %f GFLOPS\n", gflops);

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

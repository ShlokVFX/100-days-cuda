#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1024*1024)
#define ITERATIONS 100

// Naïve kernel: performs 100 iterations (with branch divergence) and then replicates the sum eight times.
__global__ void naiveKernel(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float sum = 0.0f;
        // Branch divergence: even indices use multiplication; odd indices use addition.
        if (idx % 2 == 0) {
            for (int i = 0; i < ITERATIONS; i++) {
                sum += a[idx] * b[idx];
            }
        } else {
            for (int i = 0; i < ITERATIONS; i++) {
                sum += a[idx] + b[idx];
            }
        }
        // Replicate the result eight times (i.e. multiply by 8)
        float temp1 = sum;
        float temp2 = sum;
        float temp3 = sum;
        float temp4 = sum;
        float temp5 = sum;
        float temp6 = sum;
        float temp7 = sum;
        float temp8 = sum;
        c[idx] = temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7 + temp8;
    }
}

// Warptiled kernel: unrolls the loop by a factor of 8 into independent accumulators.
// This increases ILP and helps hide long scoreboard stalls.
// It computes the per-thread “unit” value once (using a flag to choose between multiply or add)
// then performs ITERATIONS iterations (with remainder handled if ITERATIONS isn't divisible by 8)
// and finally replicates the result by multiplying by 8.
__global__ void warptiledKernel(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float valA = a[idx];
        float valB = b[idx];
        // Compute the per-iteration unit based on thread index parity.
        // For even: use a[idx]*b[idx]; for odd: use a[idx]+b[idx].
        float unit = (idx % 2 == 0) ? (valA * valB) : (valA + valB);

        // Unroll the 100 iterations by a factor of 8 using 8 independent accumulators.
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;
        int unrollFactor = 8;
        int iter = ITERATIONS / unrollFactor;  // integer division
        for (int i = 0; i < iter; i++) {
            sum0 += unit;
            sum1 += unit;
            sum2 += unit;
            sum3 += unit;
            sum4 += unit;
            sum5 += unit;
            sum6 += unit;
            sum7 += unit;
        }
        // Handle remaining iterations if ITERATIONS is not a multiple of 8.
        int remainder = ITERATIONS - iter * unrollFactor;
        float sumR = 0.0f;
        for (int i = 0; i < remainder; i++) {
            sumR += unit;
        }
        float sum = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 + sumR;
        // Replicate the result eight times.
        c[idx] = sum * 8.0f;
    }
}

int main() {
    int n = N;
    size_t size = n * sizeof(float);

    // Allocate host arrays.
    float *h_a = (float*) malloc(size);
    float *h_b = (float*) malloc(size);
    float *h_c = (float*) malloc(size);

    // Initialize host arrays.
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Allocate device memory.
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy data from host to device.
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // --------------------------
    // Run the naïve kernel.
    // --------------------------
    naiveKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    printf("Naïve Kernel Results (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("c[%d] = %f\n", i, h_c[i]);
    }

    // --------------------------
    // Run the warptiled kernel.
    // --------------------------
    warptiledKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    printf("\nWarptiled Kernel Results (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("c[%d] = %f\n", i, h_c[i]);
    }

    // Cleanup.
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}

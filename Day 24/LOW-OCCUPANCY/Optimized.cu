#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1024*1024)
#define ITERATIONS 100

// Naïve kernel: performs 100 iterations and replicates the sum eight times,
// suffering from branch divergence and high register pressure.
__global__ void naiveKernel(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float sum = 0.0f;
        // Branch divergence: even threads multiply, odd threads add.
        if (idx % 2 == 0) {
            for (int i = 0; i < ITERATIONS; i++) {
                sum += a[idx] * b[idx];
            }
        } else {
            for (int i = 0; i < ITERATIONS; i++) {
                sum += a[idx] + b[idx];
            }
        }
        // Replicate the sum eight times (increasing register pressure)
        // which effectively multiplies the sum by 8.
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

// Fully optimized kernel: since the iterative loop is a constant accumulation,
// we can replace it with a single multiplication. This kernel avoids loops,
// branch divergence, and extra temporary variables.
__global__ void fullyOptimizedKernel(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // Load from global memory once.
        float valA = a[idx];
        float valB = b[idx];
        // Compute a flag: 1.0 if even, 0.0 if odd.
        float flag = (idx % 2 == 0) ? 1.0f : 0.0f;
        // Using the flag, choose between multiplication or addition.
        // This is equivalent to:
        //   if (even) result = a[idx]*b[idx]
        //   else        result = a[idx]+b[idx]
        float result = flag * (valA * valB) + (1.0f - flag) * (valA + valB);
        // Each kernel originally loops 100 iterations and then replicates 8 times,
        // so the final answer is simply result * 800.
        c[idx] = result * 800.0f;
    }
}

int main() {
    int n = N;
    size_t size = n * sizeof(float);

    // Allocate host memory.
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

    // (Optional) Copy results back to host for correctness check.
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    printf("Naïve Kernel Results (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
         printf("c[%d] = %f\n", i, h_c[i]);
    }

    // --------------------------
    // Run the fully optimized kernel.
    // --------------------------
    fullyOptimizedKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    printf("\nFully Optimized Kernel Results (first 10 elements):\n");
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

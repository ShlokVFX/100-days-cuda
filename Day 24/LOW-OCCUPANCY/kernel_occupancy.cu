#include <stdio.h>
#include <cuda.h>
#define N (1024*1024)

// Naïve kernel: suffers from branch divergence and excessive register use.
__global__ void naiveKernel(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float sum = 0.0f;
        // Branch divergence: even and odd indices follow different loops.
        if (idx % 2 == 0) {
            for (int i = 0; i < 100; i++) {
                sum += a[idx] * b[idx];
            }
        } else {
            for (int i = 0; i < 100; i++) {
                sum += a[idx] + b[idx];
            }
        }
        // Unnecessary temporary variables increase register usage.
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

// Optimized kernel: removes branch divergence and reduces register pressure.
__global__ void optimizedKernel(float *a, float *b, float *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // Load once from global memory.
        float valA = a[idx];
        float valB = b[idx];
        // Precompute both possibilities.
        float prod = valA * valB;
        float add  = valA + valB;
        // Remove branch divergence by using a simple flag mask.
        int flag = (idx % 2 == 0);
        float multiplier = (float) flag;  // 1.0 for even, 0.0 for odd.
        float sum = 0.0f;
        for (int i = 0; i < 100; i++) {
            // Compute both options and select using the multiplier.
            sum += multiplier * prod + (1.0f - multiplier) * add;
        }
        // Write result without extra temporaries.
        c[idx] = sum;
    }
}

int main() {
    int n = N;
    size_t size = n * sizeof(float);
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
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // Copy data to device.
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    // --------------------------
    // Run the naïve kernel
    // --------------------------
    naiveKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    
    // (Optional) Copy results back to host for correctness check.
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // --------------------------
    // Run the optimized kernel
    // --------------------------
    optimizedKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    
    // (Optional) Copy results back to host for correctness check.
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // Cleanup.
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}

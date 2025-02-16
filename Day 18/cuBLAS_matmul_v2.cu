#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <assert.h>
#include <cmath>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>

#define EPSILON 1.0e-2

//verify result
void verify_result(float *a, float *b, float *c, int n) {
    float temp;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            temp = 0;
            for (int k = 0; k < n; k++) {
                temp += a[k * n + i] * b[j * n + k];  //column major
            }
            assert(fabs(c[j * n + i] - temp) < EPSILON);
        }
    }
}

int main() {
    //declare variables
    int n = 4; // Keeping small for readable output
    size_t bytes = n * n * sizeof(float);
    
    //declare pointers
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    //allocate memory
    h_a = (float *)malloc(bytes);
    h_b = (float *)malloc(bytes);
    h_c = (float *)malloc(bytes);
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    //set seed
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

    //generate random numbers
    curandGenerateUniform(prng, d_a, n * n);
    curandGenerateUniform(prng, d_b, n * n);

    //copy data back to host to print input matrices
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);
    
    printf("\nMatrix Size: %d x %d\n", n, n);
    
    printf("\nMatrix A:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.4f ", h_a[i * n + j]);
        }
        printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.4f ", h_b[i * n + j]);
        }
        printf("\n");
    }

    //create cublas handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    //scale factor
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //record start time
    cudaEventRecord(start);

    //launch kernel
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);

    //record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    //calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double seconds = milliseconds / 1000.0;
    double gflops = (2.0 * pow(n, 3)) / (seconds * 1e9);

    //copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("\nMatrix C (Result):\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.4f ", h_c[i * n + j]);
        }
        printf("\n");
    }

    printf("\n🚀 Performance Metrics:\n");
    printf(" - Execution Time: %.6f ms\n", milliseconds);
    printf(" - GFLOPS: %.6f\n", gflops);

    //verify result
    verify_result(h_a, h_b, h_c, n);

    //free memory
    cublasDestroy(handle);
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\nCOMPLETED SUCCESSFULLY\n");
    return 0;
}

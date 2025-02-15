#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>  // For fabs()

#define EPSILON 1e-5  // Small tolerance for floating-point comparisons

// Function to check cuBLAS API call results
#define CHECK_CUBLAS(call) { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error at %s:%d, error code: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
}

// Function to verify results
void verify_result(float* a, float* b, float* c, float factor, int n) {
    for (int i = 0; i < n; i++) {
        float expected = factor * a[i] + b[i];  // Correct formula
        if (fabs(c[i] - expected) > EPSILON) {
            printf("Mismatch at index %d: got %f, expected %f (factor = %f, a[i] = %f, b[i] = %f)\n",
                   i, c[i], expected, factor, a[i], b[i]);
            assert(fabs(c[i] - expected) < EPSILON);  // Assertion with tolerance
        }
    }
    printf("Verification passed!\n");
}

int main() {
    int n = 1 << 10;  // 1024 elements
    size_t bytes = n * sizeof(float);
    float factor = 2.0f;  // Scaling factor for SAXPY

    // Allocate host memory
    float *a = (float*) malloc(bytes);
    float *b = (float*) malloc(bytes);
    float *c = (float*) malloc(bytes);

    // Initialize host arrays with random values
    for (int i = 0; i < n; i++) {
        a[i] = (float)(rand() % 100);  // Corrected type casting
        b[i] = (float)(rand() % 100);
    }

    // Debugging: Print first few values before SAXPY
    printf("Factor: %f\n", factor);
    for (int i = 0; i < 5; i++) {
        printf("Before SAXPY: a[%d] = %f, b[%d] = %f\n", i, a[i], i, b[i]);
    }

    // Allocate device memory
    float *d_a, *d_b;
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);

    // Copy data from host to device
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Perform SAXPY operation: d_b = factor * d_a + d_b
    CHECK_CUBLAS(cublasSaxpy(handle, n, &factor, d_a, 1, d_b, 1));

    // Debugging: Check device values after SAXPY
    float* temp_b = (float*)malloc(bytes);
    cudaMemcpy(temp_b, d_b, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 5; i++) {
        printf("After SAXPY (Device): b[%d] = %f\n", i, temp_b[i]);
    }
    free(temp_b);

    // Copy result back to host
    cudaMemcpy(c, d_b, bytes, cudaMemcpyDeviceToHost);

    // Verify the result
    verify_result(a, b, c, factor, n);

    // Clean up
    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    free(a);
    free(b);
    free(c);

    printf("Program completed successfully!\n");
    return 0;
}

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

// Function to verify results and compute error
void verify_result(float* a, float* b, float* c, float factor, int n) {
    double sum_abs_error = 0.0;
    double max_relative_error = 0.0;

    for (int i = 0; i < n; i++) {
        float expected = factor * a[i] + b[i];  
        float abs_error = fabs(c[i] - expected);
        float relative_error = abs_error / (fabs(expected) + 1e-8);

        sum_abs_error += abs_error;
        if (relative_error > max_relative_error) {
            max_relative_error = relative_error;
        }

        if (fabs(c[i] - expected) > EPSILON) {
            printf("Mismatch at index %d: got %f, expected %f (factor = %f, a[i] = %f, b[i] = %f)\n",
                   i, c[i], expected, factor, a[i], b[i]);
        }
    }

    double mean_absolute_error = sum_abs_error / n;
    printf("\n🔍 Error Metrics:\n");
    printf(" - Mean Absolute Error (MAE): %e\n", mean_absolute_error);
    printf(" - Max Relative Error: %e\n", max_relative_error);
}

int main() {
    int n = 1 << 20;  // 1,048,576 elements
    size_t bytes = n * sizeof(float);
    float factor = 2.0f;

    // Allocate host memory
    float *a = (float*) malloc(bytes);
    float *b = (float*) malloc(bytes);
    float *c = (float*) malloc(bytes);

    // Initialize host arrays with random values
    for (int i = 0; i < n; i++) {
        a[i] = (float)(rand() % 100);
        b[i] = (float)(rand() % 100);
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

    // CUDA events for performance measurement
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Perform SAXPY operation: d_b = factor * d_a + d_b
    CHECK_CUBLAS(cublasSaxpy(handle, n, &factor, d_a, 1, d_b, 1));

    // Record stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Compute elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double seconds = milliseconds / 1000.0;
    double gflops = (2.0 * n) / (seconds * 1e9);

    printf("\n🚀 Performance Metrics:\n");
    printf(" - Execution Time: %.6f ms\n", milliseconds);
    printf(" - GFLOPS: %.6f\n", gflops);

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
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n✅ Program completed successfully!\n");
    return 0;
}

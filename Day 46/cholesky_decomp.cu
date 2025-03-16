#include <stdio.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#define N 3

int main() {
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    float h_A[N * N] = { 4.0f, 12.0f, -16.0f,
                        12.0f, 37.0f, -43.0f,
                        -16.0f, -43.0f, 98.0f };

    printf("Input Matrix A (3x3):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", h_A[i * N + j]);
        }
        printf("\n");
    }

    printf("\nFormula: A = L * L^T\n");
    printf("Extracting Lower Triangular Matrix L\n\n");

    float h_L[N * N];
    float *d_A;
    int *devInfo;
    int lwork;
    float *d_work;

    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&devInfo, sizeof(int));
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);

    cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, N, d_A, N, &lwork);
    cudaMalloc((void**)&d_work, lwork * sizeof(float));

    cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_LOWER, N, d_A, N, d_work, lwork, devInfo);
    cudaMemcpy(h_L, d_A, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output Matrix L (Lower Triangular):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i >= j)
                printf("%f ", h_L[i * N + j]);
            else
                printf("0.000000 ");
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_work);
    cudaFree(devInfo);
    cusolverDnDestroy(handle);

    return 0;
}

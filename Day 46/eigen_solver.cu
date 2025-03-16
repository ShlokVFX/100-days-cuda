#include <stdio.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#define N 3

int main() {
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    float h_A[N * N] = { 3.0f, 1.0f, 0.0f,
                         1.0f, 2.0f, 0.0f,
                         0.0f, 0.0f, 1.0f };

    printf("Input Matrix A (3x3):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", h_A[i * N + j]);
        }
        printf("\n");
    }

    printf("\nCalculating Eigenvalues & Eigenvectors using cuSolver\n\n");

    float h_W[N];  // Eigenvalues
    float *d_A, *d_W;
    int *devInfo;
    int lwork;
    float *d_work;

    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_W, N * sizeof(float));
    cudaMalloc((void**)&devInfo, sizeof(int));
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);

    cusolverDnSsyevd_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                                N, d_A, N, d_W, &lwork);
    cudaMalloc((void**)&d_work, lwork * sizeof(float));

    cusolverDnSsyevd(handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                     N, d_A, N, d_W, d_work, lwork, devInfo);
    cudaMemcpy(h_W, d_W, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_A, d_A, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Eigenvalues:\n");
    for (int i = 0; i < N; i++) {
        printf("%f\n", h_W[i]);
    }

    printf("\nEigenvectors (Column-wise):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", h_A[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_W);
    cudaFree(d_work);
    cudaFree(devInfo);
    cusolverDnDestroy(handle);

    return 0;
}

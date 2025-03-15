#include <stdio.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#define M 3
#define N 2

int main() {
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    float h_A[M * N] = { 1.0f, 2.0f,
                         4.0f, 5.0f,
                         7.0f, 8.0f };

    printf("Input Matrix A (3x2):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", h_A[i * N + j]);
        }
        printf("\n");
    }

    printf("\nFormula: A = Q * R\n");
    printf("We are only extracting matrix R (Upper Triangular Matrix)\n\n");

    float h_R[M * N];
    float *d_A, *d_tau, *d_work;
    int *devInfo;
    int lwork;

    cudaMalloc((void**)&d_A, M * N * sizeof(float));
    cudaMalloc((void**)&d_tau, N * sizeof(float));
    cudaMalloc((void**)&devInfo, sizeof(int));
    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cusolverDnSgeqrf_bufferSize(handle, M, N, d_A, M, &lwork);
    cudaMalloc((void**)&d_work, lwork * sizeof(float));
    cusolverDnSgeqrf(handle, M, N, d_A, M, d_tau, d_work, lwork, devInfo);
    cudaMemcpy(h_R, d_A, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output Matrix R (Upper Triangular):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (i <= j)
                printf("%f ", h_R[i * N + j]);
            else
                printf("0.000000 ");
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_tau);
    cudaFree(d_work);
    cudaFree(devInfo);
    cusolverDnDestroy(handle);

    return 0;
}

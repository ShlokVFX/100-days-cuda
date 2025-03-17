#include <stdio.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#define M 3
#define N 2

int main() {
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    float h_A[M * N] = { 3.0f, 1.0f,
                         1.0f, 3.0f,
                         0.0f, 1.0f };

    printf("Input Matrix A (3x2):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", h_A[i * N + j]);
        }
        printf("\n");
    }

    printf("\nCalculating SVD: A = U * S * V^T\n\n");

    float h_S[N];          
    float h_U[M * M];      
    float h_VT[N * N];      
    float *d_A, *d_S, *d_U, *d_VT;
    int *devInfo;
    int lwork;
    float *d_work;
    float *rwork = NULL;

    cudaMalloc((void**)&d_A, M * N * sizeof(float));
    cudaMalloc((void**)&d_S, N * sizeof(float));
    cudaMalloc((void**)&d_U, M * M * sizeof(float));
    cudaMalloc((void**)&d_VT, N * N * sizeof(float));
    cudaMalloc((void**)&devInfo, sizeof(int));

    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);

    cusolverDnSgesvd_bufferSize(handle, M, N, &lwork);
    cudaMalloc((void**)&d_work, lwork * sizeof(float));

    char jobu = 'A'; 
    char jobvt = 'A'; 

    cusolverDnSgesvd(handle, jobu, jobvt, M, N, d_A, M, d_S, d_U, M, d_VT, N, d_work, lwork, rwork, devInfo);

    cudaMemcpy(h_S, d_S, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_U, d_U, M * M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_VT, d_VT, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Singular Values (S):\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", h_S[i]);
    }
    printf("\n\n");

    printf("Left Singular Vectors (U):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            printf("%f ", h_U[i * M + j]);
        }
        printf("\n");
    }

    printf("\nRight Singular Vectors (V^T):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", h_VT[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_S);
    cudaFree(d_U);
    cudaFree(d_VT);
    cudaFree(d_work);
    cudaFree(devInfo);
    cusolverDnDestroy(handle);

    return 0;
}

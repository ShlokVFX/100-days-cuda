#include <stdio.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#define N 3  // Size of the matrix

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        printf("%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void checkCusolverError(cusolverStatus_t err, const char* msg) {
    if (err != CUSOLVER_STATUS_SUCCESS) {
        printf("%s\n", msg);
        exit(EXIT_FAILURE);
    }
}

int main() {

    float h_A[N * N] = { 1.0f, 2.0f, 3.0f,
                         4.0f, 5.0f, 6.0f,
                         7.0f, 8.0f, 10.0f };
    float h_B[N] = { 1.0f, 2.0f, 3.0f };
    
    float *d_A, *d_B;
    int *d_info, *d_pivot;
    
    checkCudaError(cudaMalloc((void**)&d_A, N * N * sizeof(float)), "Failed to allocate d_A");
    checkCudaError(cudaMalloc((void**)&d_B, N * sizeof(float)), "Failed to allocate d_B");
    checkCudaError(cudaMalloc((void**)&d_info, sizeof(int)), "Failed to allocate d_info");
    checkCudaError(cudaMalloc((void**)&d_pivot, N * sizeof(int)), "Failed to allocate d_pivot");

    checkCudaError(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy A to device");
    checkCudaError(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy B to device");


    cusolverDnHandle_t handle;
    checkCusolverError(cusolverDnCreate(&handle), "Failed to create cuSolver handle");

    // Workspace query
    int work_size = 0;
    checkCusolverError(cusolverDnSgetrf_bufferSize(handle, N, N, d_A, N, &work_size),
                       "Failed to query buffer size");

    float *d_work;
    checkCudaError(cudaMalloc((void**)&d_work, work_size * sizeof(float)), "Failed to allocate d_work");

    // LU Factorization
    checkCusolverError(cusolverDnSgetrf(handle, N, N, d_A, N, d_work, d_pivot, d_info),
                       "Failed to perform LU factorization");

    // Solve AX = B
    checkCusolverError(cusolverDnSgetrs(handle, CUBLAS_OP_N, N, 1, d_A, N, d_pivot, d_B, N, d_info),
                       "Failed to solve linear system");

    checkCudaError(cudaMemcpy(h_B, d_B, N * sizeof(float), cudaMemcpyDeviceToHost),
                   "Failed to copy result to host");

    printf("Solution X:\n");
    for (int i = 0; i < N; i++) {
        printf("%f\n", h_B[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_work);
    cudaFree(d_info);
    cudaFree(d_pivot);
    cusolverDnDestroy(handle);

    return 0;
}

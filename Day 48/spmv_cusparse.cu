#include <iostream>
#include <cusparse.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                   \
    {                                                                      \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << '\n'; \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    }

#define CHECK_CUSPARSE(call)                                              \
    {                                                                     \
        cusparseStatus_t status = call;                                   \
        if (status != CUSPARSE_STATUS_SUCCESS) {                          \
            std::cerr << "cuSPARSE error: " << status << '\n';            \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

int main() {
    const int A_num_rows = 4;
    const int A_num_cols = 4;
    const int A_nnz = 7;
    const int x_size = A_num_cols;
    const int y_size = A_num_rows;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    int hA_rowOffsets[] = {0, 1, 2, 5, 7};
    int hA_columns[] = {0, 1, 0, 1, 2, 0, 3};
    float hA_values[] = {0.5f, 1.0f, 0.2f, 0.3f, 1.2f, 0.4f, 1.1f};
    float hX[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float hY[] = {0.0f, 0.0f, 0.0f, 0.0f};

    int *dA_rowOffsets, *dA_columns;
    float *dA_values, *dX, *dY;
    CHECK_CUDA(cudaMalloc((void**)&dA_rowOffsets, (A_num_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dA_columns, A_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dA_values, A_nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dX, x_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dY, y_size * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA_rowOffsets, hA_rowOffsets, (A_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dX, hX, x_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dY, hY, y_size * sizeof(float), cudaMemcpyHostToDevice));

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                     dA_rowOffsets, dA_columns, dA_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, x_size, dX, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, y_size, dY, CUDA_R_32F));

    size_t bufferSize = 0;
    void *dBuffer = nullptr;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY,
         CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY,
         CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);

    CHECK_CUDA(cudaMemcpy(hY, dY, y_size * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Resultant vector y:\n";
    for (int i = 0; i < y_size; i++) {
        std::cout << hY[i] << " ";
    }
    std::cout << std::endl;

    CHECK_CUDA(cudaFree(dA_values));
    CHECK_CUDA(cudaFree(dA_columns));
    CHECK_CUDA(cudaFree(dA_rowOffsets));
    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));
    CHECK_CUDA(cudaFree(dBuffer));

    CHECK_CUSPARSE(cusparseDestroy(handle));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));

    return 0;
}

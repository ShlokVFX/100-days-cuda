#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>

#define PRUNE_THRESHOLD 0.0001f
#define N 4096

using namespace std;

__global__ void denseCompressionKernel(const float* input, float* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = roundf(input[idx] * scale) / scale;
    }
}

__global__ void pruneKernelOptimized(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < size; i += stride) {
        if (fabsf(data[i]) < PRUNE_THRESHOLD)
            data[i] = 0.0f;
    }
}

void generateRandomData(float* data, int size, float sparsity = 0.1f) {
    for (int i = 0; i < size; i++) {
        data[i] = (rand() % 10 < (sparsity * 10)) ? static_cast<float>(rand() % 100) : 0.0f;
    }
}

void compressSparseCSR(float* d_denseMatrix, int rows, int cols, cusparseHandle_t handle, cudaStream_t stream) {
    cusparseSetStream(handle, stream);
    cusparseDnMatDescr_t matA = nullptr;
    cusparseSpMatDescr_t matB = nullptr;
    cusparseCreateDnMat(&matA, rows, cols, rows, d_denseMatrix, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    int *d_csrRowPtr = nullptr, *d_csrColInd = nullptr;
    float *d_csrVal = nullptr;
    cudaMalloc(&d_csrRowPtr, (rows + 1) * sizeof(int));
    cudaMalloc(&d_csrColInd, rows * cols * sizeof(int));
    cudaMalloc(&d_csrVal, rows * cols * sizeof(float));
    cudaMemsetAsync(d_csrRowPtr, 0, (rows + 1) * sizeof(int), stream);
    cusparseCreateCsr(&matB, rows, cols, 0, d_csrRowPtr, d_csrColInd, d_csrVal,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    pruneKernelOptimized<<<(rows * cols + 255) / 256, 256, 0, stream>>>(d_denseMatrix, rows * cols);
    cudaStreamSynchronize(stream);
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    cusparseDenseToSparse_bufferSize(handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
    cusparseDenseToSparse_analysis(handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer);
    cusparseDenseToSparse_convert(handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer);
    int nnz;
    cudaMemcpyAsync(&nnz, d_csrRowPtr + rows, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cout << "Sparse Compression (CSR) - Non-Zero Count: " << nnz << "\n";
    cudaFree(dBuffer);
    cusparseDestroyDnMat(matA);
    cusparseDestroySpMat(matB);
    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColInd);
    cudaFree(d_csrVal);
}

int main() {
    srand(0);
    float* h_denseMatrix = nullptr;
    cudaMallocHost(&h_denseMatrix, N * N * sizeof(float));
    generateRandomData(h_denseMatrix, N * N);
    float* d_denseMatrix = nullptr;
    cudaMalloc(&d_denseMatrix, N * N * sizeof(float));
    cudaMemcpy(d_denseMatrix, h_denseMatrix, N * N * sizeof(float), cudaMemcpyHostToDevice);
    float* d_compressedDense = nullptr;
    cudaMalloc(&d_compressedDense, N * N * sizeof(float));
    int initialNNZ = 0;
    for (int i = 0; i < N * N; i++) {
        if (h_denseMatrix[i] != 0.0f)
            initialNNZ++;
    }
    cout << "Initial Non-Zero Count: " << initialNNZ << "\n";
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaStream_t sparseStream;
    cudaStreamCreate(&sparseStream);
    cudaEventRecord(startEvent, 0);
    denseCompressionKernel<<<(N * N + 255) / 256, 256>>>(d_denseMatrix, d_compressedDense, 100.0f, N * N);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float denseTime = 0.0f;
    cudaEventElapsedTime(&denseTime, startEvent, stopEvent);
    cout << "Dense Compression Time: " << denseTime << " ms\n";
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseSetStream(handle, sparseStream);
    cudaEventRecord(startEvent, sparseStream);
    compressSparseCSR(d_denseMatrix, N, N, handle, sparseStream);
    cudaEventRecord(stopEvent, sparseStream);
    cudaStreamSynchronize(sparseStream);
    float sparseTime = 0.0f;
    cudaEventElapsedTime(&sparseTime, startEvent, stopEvent);
    cout << "Sparse Compression Time: " << sparseTime << " ms\n";
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaStreamDestroy(sparseStream);
    cudaFree(d_denseMatrix);
    cudaFree(d_compressedDense);
    cudaFreeHost(h_denseMatrix);
    cusparseDestroy(handle);
    return 0;
}

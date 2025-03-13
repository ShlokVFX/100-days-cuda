#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <chrono>

#define M 1024
#define N 1024
#define K 1024

#define BLOCK_SIZE 16

__global__ void int8_gemm(int8_t* A, int8_t* B, int32_t* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        int32_t sum = 0;
        for (int i = 0; i < k; ++i) {
            sum += static_cast<int32_t>(A[row * k + i]) * static_cast<int32_t>(B[i * n + col]);
        }
        C[row * n + col] = sum;
    }
}

__global__ void fp32_gemm(float* A, float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

void quantize(float* input, int8_t* output, int size, float scale) {
    for (int i = 0; i < size; ++i) {
        output[i] = static_cast<int8_t>(roundf(input[i] * scale));
    }
}

void dequantize(int32_t* input, float* output, int size, float scale) {
    for (int i = 0; i < size; ++i) {
        output[i] = static_cast<float>(input[i]) / (scale * scale);
    }
}

float compute_mae(float* A, float* B, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += std::abs(A[i] - B[i]);
    }
    return sum / size;
}

int main() {
    float *A_fp32, *B_fp32, *C_fp32, *C_fp32_ref;
    int8_t *A_int8, *B_int8;
    int32_t *C_int32;

    A_fp32 = (float*)malloc(M * K * sizeof(float));
    B_fp32 = (float*)malloc(K * N * sizeof(float));
    C_fp32 = (float*)malloc(M * N * sizeof(float));
    C_fp32_ref = (float*)malloc(M * N * sizeof(float));

    A_int8 = (int8_t*)malloc(M * K * sizeof(int8_t));
    B_int8 = (int8_t*)malloc(K * N * sizeof(int8_t));
    C_int32 = (int32_t*)malloc(M * N * sizeof(int32_t));

    for (int i = 0; i < M * K; ++i) A_fp32[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) B_fp32[i] = static_cast<float>(rand()) / RAND_MAX;

    float scale = 127.0f;

    quantize(A_fp32, A_int8, M * K, scale);
    quantize(B_fp32, B_int8, K * N, scale);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    int8_t *d_A, *d_B;
    int32_t *d_C;

    cudaMalloc(&d_A, M * K * sizeof(int8_t));
    cudaMalloc(&d_B, K * N * sizeof(int8_t));
    cudaMalloc(&d_C, M * N * sizeof(int32_t));

    cudaMemcpyAsync(d_A, A_int8, M * K * sizeof(int8_t), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B, B_int8, K * N * sizeof(int8_t), cudaMemcpyHostToDevice, stream2);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);
    int8_gemm<<<grid, block, 0, stream1>>>(d_A, d_B, d_C, M, N, K);
    cudaStreamEndCapture(stream1, &graph);

    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    auto start = std::chrono::high_resolution_clock::now();
    cudaGraphLaunch(graphExec, stream1);
    cudaStreamSynchronize(stream1);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration_ms = end - start;
    float gflops = (2.0f * M * N * K) / (duration_ms.count() * 1e6);

    cudaMemcpy(C_int32, d_C, M * N * sizeof(int32_t), cudaMemcpyDeviceToHost);

    std::cout << "INT8 GEMM - Time: " << duration_ms.count() << " ms, GFLOPS: " << gflops << std::endl;

    dequantize(C_int32, C_fp32, M * N, scale);

    start = std::chrono::high_resolution_clock::now();
    fp32_gemm<<<grid, block>>>(A_fp32, B_fp32, C_fp32_ref, M, N, K);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();

    duration_ms = end - start;
    gflops = (2.0f * M * N * K) / (duration_ms.count() * 1e6);

    std::cout << "FP32 GEMM - Time: " << duration_ms.count() << " ms, GFLOPS: " << gflops << std::endl;

    float mae = compute_mae(C_fp32, C_fp32_ref, M * N);
    std::cout << "Mean Absolute Error (MAE) between FP32 and INT8 GEMM: " << mae << std::endl;

    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graphExec);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A_fp32);
    free(B_fp32);
    free(C_fp32);
    free(C_fp32_ref);
    free(A_int8);
    free(B_int8);
    free(C_int32);

    return 0;
}

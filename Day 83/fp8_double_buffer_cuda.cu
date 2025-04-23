// fp8_double_buffer_cuda.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <chrono>
#include <random>

#define TILE_M 128
#define TILE_N 128
#define TILE_K 32
#define VECTOR_SIZE 4

// Device-side version (GPU)
__device__ float fp8_dequantize(uint8_t val, float scale) {
    return (static_cast<float>(val) / 127.0f - 1.0f) * scale;
}

__device__ uint8_t fp8_quantize(float val, float scale) {
    float clamped = fminf(fmaxf(val / scale, -1.0f), 1.0f);
    return static_cast<uint8_t>((clamped + 1.0f) * 127.0f);
}

// Host-side version (CPU)
float fp8_quantize_host(float val, float scale) {
    float clamped = std::min(std::max(val / scale, -1.0f), 1.0f);
    return static_cast<uint8_t>((clamped + 1.0f) * 127.0f);
}

__global__ void fp8_gemm_kernel(const uint8_t *A, const uint8_t *B, const float *A_scale, const float *B_scale, float *C, int M, int N, int K) {
    __shared__ uint8_t As[2][TILE_K][TILE_M];
    __shared__ uint8_t Bs[2][TILE_K][TILE_N];
    __shared__ float AScales[2][TILE_M];
    __shared__ float BScales[2][TILE_N];

    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    float acc = 0.0f;
    int cur = 0, nxt = 1;

    for (int t = 0; t < K; t += TILE_K) {
        if (row < M && (t + threadIdx.x) < K)
            As[cur][threadIdx.x][threadIdx.y] = A[row * K + (t + threadIdx.x)];
        if (col < N && (t + threadIdx.y) < K)
            Bs[cur][threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];

        if (threadIdx.y == 0 && (blockIdx.y * TILE_M + threadIdx.x) < M)
            AScales[cur][threadIdx.x] = A_scale[blockIdx.y * TILE_M + threadIdx.x];
        if (threadIdx.x == 0 && (blockIdx.x * TILE_N + threadIdx.y) < N)
            BScales[cur][threadIdx.y] = B_scale[blockIdx.x * TILE_N + threadIdx.y];

        __syncthreads();

        for (int k = 0; k < TILE_K; ++k) {
            float a_val = fp8_dequantize(As[cur][k][threadIdx.y], AScales[cur][threadIdx.y]);
            float b_val = fp8_dequantize(Bs[cur][k][threadIdx.x], BScales[cur][threadIdx.x]);
            acc += a_val * b_val;
        }

        __syncthreads();
        cur ^= 1; nxt ^= 1;
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " M N K seed\n";
        return 1;
    }
    int M = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    int K = std::stoi(argv[3]);
    int seed = std::stoi(argv[4]);

    size_t sizeA = size_t(M) * K;
    size_t sizeB = size_t(K) * N;
    size_t sizeC = size_t(M) * N;

    uint8_t *h_A = new uint8_t[sizeA];
    uint8_t *h_B = new uint8_t[sizeB];
    float *h_A_scale = new float[M];
    float *h_B_scale = new float[N];
    float *h_C = new float[sizeC];

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < sizeA; ++i) h_A[i] = fp8_quantize_host(dist(rng), 1.0f);
    for (size_t i = 0; i < sizeB; ++i) h_B[i] = fp8_quantize_host(dist(rng), 1.0f);
    
    for (int i = 0; i < M; ++i) h_A_scale[i] = 1.0f;
    for (int i = 0; i < N; ++i) h_B_scale[i] = 1.0f;

    uint8_t *d_A, *d_B;
    float *d_A_scale, *d_B_scale, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_A_scale, M * sizeof(float));
    cudaMalloc(&d_B_scale, N * sizeof(float));
    cudaMalloc(&d_C, sizeC * sizeof(float));

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_scale, h_A_scale, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_scale, h_B_scale, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    auto t0 = std::chrono::high_resolution_clock::now();
    fp8_gemm_kernel<<<blocks, threads>>>(d_A, d_B, d_A_scale, d_B_scale, d_C, M, N, K);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    double micros = std::chrono::duration<double, std::micro>(t1 - t0).count();
    std::cout << "M=" << M << " N=" << N << " K=" << K << " seed=" << seed << " -> " << micros << " us\n";

    cudaMemcpy(h_C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_A_scale); cudaFree(d_B_scale); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_A_scale; delete[] h_B_scale; delete[] h_C;
    return 0;
}

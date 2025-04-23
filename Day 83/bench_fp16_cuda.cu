// bench_fp16_cuda.cu
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>

#define TILE_SIZE 16

__global__ void gemm_fp16_kernel(__half *A, __half *B, __half *C, int M, int N, int K) {
    __shared__ __half As[2][TILE_SIZE][TILE_SIZE];
    __shared__ __half Bs[2][TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __half accum = __float2half(0.0f);
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    int cur = 0, nxt = 1;

    // preload first tile
    int t = 0;
    int aCol = t * TILE_SIZE + threadIdx.x;
    int bRow = t * TILE_SIZE + threadIdx.y;
    As[cur][threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row*K + aCol] : __float2half(0.0f);
    Bs[cur][threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow*N + col] : __float2half(0.0f);
    __syncthreads();

    for (t = 0; t < numTiles; ++t) {
        if (t + 1 < numTiles) {
            int naCol = (t+1)*TILE_SIZE + threadIdx.x;
            int nbRow = (t+1)*TILE_SIZE + threadIdx.y;
            As[nxt][threadIdx.y][threadIdx.x] = (row < M && naCol < K) ? A[row*K + naCol] : __float2half(0.0f);
            Bs[nxt][threadIdx.y][threadIdx.x] = (nbRow < K && col < N) ? B[nbRow*N + col] : __float2half(0.0f);
        }
        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) {
            accum = __hadd(accum, __hmul(As[cur][threadIdx.y][i], Bs[cur][i][threadIdx.x]));
        }
        __syncthreads();
        cur ^= 1; nxt ^= 1;
    }

    if (row < M && col < N) {
        C[row*N + col] = accum;
    }
}

inline void check_cuda(cudaError_t err, const char* action) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error during " << action << ": " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " M N K seed\n";
        return 1;
    }
    int M = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    int K = std::stoi(argv[3]);
    int seed = std::stoi(argv[4]);

    size_t sizeA = size_t(M)*K;
    size_t sizeB = size_t(K)*N;
    size_t sizeC = size_t(M)*N;

    __half *h_A = new __half[sizeA];
    __half *h_B = new __half[sizeB];
    __half *h_C = new __half[sizeC];

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < sizeA; ++i) h_A[i] = __float2half(dist(rng));
    for (size_t i = 0; i < sizeB; ++i) h_B[i] = __float2half(dist(rng));

    __half *d_A, *d_B, *d_C;
    check_cuda(cudaMalloc(&d_A, sizeA * sizeof(__half)), "alloc d_A");
    check_cuda(cudaMalloc(&d_B, sizeB * sizeof(__half)), "alloc d_B");
    check_cuda(cudaMalloc(&d_C, sizeC * sizeof(__half)), "alloc d_C");

    check_cuda(cudaMemcpy(d_A, h_A, sizeA*sizeof(__half), cudaMemcpyHostToDevice), "copy A");
    check_cuda(cudaMemcpy(d_B, h_B, sizeB*sizeof(__half), cudaMemcpyHostToDevice), "copy B");

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N+TILE_SIZE-1)/TILE_SIZE, (M+TILE_SIZE-1)/TILE_SIZE);

    // Warm-up
    gemm_fp16_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    check_cuda(cudaDeviceSynchronize(), "warmup sync");

    // Timed run
    auto t0 = std::chrono::high_resolution_clock::now();
    gemm_fp16_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    check_cuda(cudaDeviceSynchronize(), "timed sync");
    auto t1 = std::chrono::high_resolution_clock::now();

    double micros = std::chrono::duration<double, std::micro>(t1 - t0).count();
    std::cout << "M=" << M << " N=" << N << " K=" << K
              << " seed=" << seed << " -> " << micros << " us\n";

    check_cuda(cudaMemcpy(h_C, d_C, sizeC*sizeof(__half), cudaMemcpyDeviceToHost), "copy C back");
    check_cuda(cudaFree(d_A), "free d_A");
    check_cuda(cudaFree(d_B), "free d_B");
    check_cuda(cudaFree(d_C), "free d_C");
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
}


// bench_fp16.cpp
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <iostream>
#include <chrono>
#include <random>

#define TILE_SIZE 16

// FP16 GEMM Kernel with double-buffering
__global__ void gemm_fp16_kernel(__half *A, __half *B, __half *C, int M, int N, int K) {
    __shared__ __half As[2][TILE_SIZE][TILE_SIZE];
    __shared__ __half Bs[2][TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __half accum = __half(0);
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    int cur = 0, nxt = 1;

    // preload first tile
    int t = 0;
    int aCol = t * TILE_SIZE + threadIdx.x;
    int bRow = t * TILE_SIZE + threadIdx.y;
    As[cur][threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row*K + aCol] : __half(0);
    Bs[cur][threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow*N + col] : __half(0);
    __syncthreads();

    for (t = 0; t < numTiles; ++t) {
        // preload next
        if (t + 1 < numTiles) {
            int naCol = (t+1)*TILE_SIZE + threadIdx.x;
            int nbRow = (t+1)*TILE_SIZE + threadIdx.y;
            As[nxt][threadIdx.y][threadIdx.x] = (row < M && naCol < K) ? A[row*K + naCol] : __half(0);
            Bs[nxt][threadIdx.y][threadIdx.x] = (nbRow < K && col < N) ? B[nbRow*N + col] : __half(0);
        }
        __syncthreads();
        // compute
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

inline void check_error(hipError_t err, const char* action) {
    if (err != hipSuccess) {
        std::cerr << "HIP error during " << action << ": " << hipGetErrorString(err) << std::endl;
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

    // Host buffers
    __half *h_A = new __half[sizeA];
    __half *h_B = new __half[sizeB];
    __half *h_C = new __half[sizeC];

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < sizeA; ++i) h_A[i] = __half(dist(rng));
    for (size_t i = 0; i < sizeB; ++i) h_B[i] = __half(dist(rng));

    // Device buffers
    __half *d_A, *d_B, *d_C;
    check_error(hipMalloc(&d_A, sizeA * sizeof(__half)), "alloc d_A");
    check_error(hipMalloc(&d_B, sizeB * sizeof(__half)), "alloc d_B");
    check_error(hipMalloc(&d_C, sizeC * sizeof(__half)), "alloc d_C");

    check_error(hipMemcpy(d_A, h_A, sizeA*sizeof(__half), hipMemcpyHostToDevice), "copy A");
    check_error(hipMemcpy(d_B, h_B, sizeB*sizeof(__half), hipMemcpyHostToDevice), "copy B");

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N+TILE_SIZE-1)/TILE_SIZE, (M+TILE_SIZE-1)/TILE_SIZE);

    // Warm-up
    hipLaunchKernelGGL(gemm_fp16_kernel, blocks, threads, 0, 0, d_A, d_B, d_C, M, N, K);
    hipDeviceSynchronize();

    // Timed run
    auto t0 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(gemm_fp16_kernel, blocks, threads, 0, 0, d_A, d_B, d_C, M, N, K);
    hipDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    double micros = std::chrono::duration<double, std::micro>(t1 - t0).count();
    std::cout << "M=" << M << " N=" << N << " K=" << K
              << " seed=" << seed << " -> " << micros << " us\n";

    // Cleanup
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
}


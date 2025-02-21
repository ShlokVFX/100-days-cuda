#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <mma.h>

using namespace nvcuda;

const int M = 32;
const int N = 32;
const int K = 32;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

const int M_TILES = M / WMMA_M;
const int N_TILES = N / WMMA_N;
const int K_TILES = K / WMMA_K;

__global__ void wmma_kernel(half* a, half* b, float* d, int M, int N, int K) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    const int tile_row = blockIdx.x;
    const int tile_col = blockIdx.y;

    for (int k = 0; k < K_TILES; ++k) {
        int a_row = tile_row * WMMA_M;
        int a_col = k * WMMA_K;
        const half* a_tile_ptr = a + a_row * K + a_col;
        wmma::load_matrix_sync(a_frag, a_tile_ptr, K);

        int b_row = k * WMMA_K;
        int b_col = tile_col * WMMA_N;
        const half* b_tile_ptr = b + b_row + b_col * K;
        wmma::load_matrix_sync(b_frag, b_tile_ptr, K);

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    int d_row = tile_row * WMMA_M;
    int d_col = tile_col * WMMA_N;
    float* d_tile_ptr = d + d_row * N + d_col;
    wmma::store_matrix_sync(d_tile_ptr, acc_frag, N, wmma::mem_row_major);
}

// Function to print matrices
template <typename T>
void printMatrix(const char* name, const T* matrix, int rows, int cols) {
    std::cout << name << ":\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

int main() {
    std::vector<half> host_a(M * K);
    std::vector<half> host_b(K * N);
    std::vector<float> host_d(M * N);
    std::vector<float> host_ref(M * N);

    for (int i = 0; i < M * K; ++i)
        host_a[i] = __float2half(static_cast<float>(rand() % 10) / 10.0f);

    for (int j = 0; j < N; ++j)
        for (int i = 0; i < K; ++i)
            host_b[i + j * K] = __float2half(static_cast<float>(rand() % 10) / 10.0f);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += __half2float(host_a[i * K + k]) *
                    __half2float(host_b[k + j * K]);
            }
            host_ref[i * N + j] = sum;
        }
    }

    printMatrix("Matrix A (Row-Major)", host_a.data(), M, K);
    printMatrix("Matrix B (Column-Major)", host_b.data(), K, N);

    half* device_a, * device_b;
    float* device_d;
    cudaMalloc(&device_a, M * K * sizeof(half));
    cudaMalloc(&device_b, K * N * sizeof(half));
    cudaMalloc(&device_d, M * N * sizeof(float));

    cudaMemcpy(device_a, host_a.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);

    dim3 grid(M_TILES, N_TILES);
    dim3 block(32);  // One warp per tile
    wmma_kernel<<<grid, block>>>(device_a, device_b, device_d, M, N, K);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
}

    cudaMemcpy(host_d.data(), device_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    printMatrix("Matrix D (Result from GPU)", host_d.data(), M, N);
    printMatrix("Reference Matrix (Computed on CPU)", host_ref.data(), M, N);

    float max_error = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        max_error = fmax(max_error, fabs(host_d[i] - host_ref[i]));
    }
    std::cout << "Max error: " << max_error << std::endl;

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_d);

    return 0;
}

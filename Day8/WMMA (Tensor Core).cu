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
    // Declare matrix fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // Initialize accumulator to zero
    wmma::fill_fragment(acc_frag, 0.0f);

    // Calculate tile coordinates
    const int tile_row = blockIdx.x;
    const int tile_col = blockIdx.y;

    // Accumulate matrix products for each K tile
    for (int k = 0; k < K_TILES; ++k) {
        // Load A tile (row-major)
        int a_row = tile_row * WMMA_M;
        int a_col = k * WMMA_K;
        const half* a_tile_ptr = a + a_row * K + a_col;
        wmma::load_matrix_sync(a_frag, a_tile_ptr, K);

        // Load B tile (column-major)
        int b_row = k * WMMA_K;
        int b_col = tile_col * WMMA_N;
        const half* b_tile_ptr = b + b_row + b_col * K;
        wmma::load_matrix_sync(b_frag, b_tile_ptr, K);

        // Matrix multiply-accumulate
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Store result to global memory (row-major)
    int d_row = tile_row * WMMA_M;
    int d_col = tile_col * WMMA_N;
    float* d_tile_ptr = d + d_row * N + d_col;
    wmma::store_matrix_sync(d_tile_ptr, acc_frag, N, wmma::mem_row_major);
}

int main() {
    // Host memory allocations
    std::vector<half> host_a(M * K);
    std::vector<half> host_b(K * N);  // Column-major storage
    std::vector<float> host_d(M * N);
    std::vector<float> host_ref(M * N);

    // Initialize matrices with random values
    for (int i = 0; i < M * K; ++i)
        host_a[i] = __float2half(static_cast<float>(rand() % 10) / 10.0f);

    for (int j = 0; j < N; ++j)
        for (int i = 0; i < K; ++i)
            host_b[i + j * K] = __float2half(static_cast<float>(rand() % 10) / 10.0f);

    // Compute reference on CPU
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

    // Device memory allocations
    half* device_a, * device_b;
    float* device_d;
    cudaMalloc(&device_a, M * K * sizeof(half));
    cudaMalloc(&device_b, K * N * sizeof(half));
    cudaMalloc(&device_d, M * N * sizeof(float));

    // Copy data to device
    cudaMemcpy(device_a, host_a.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);

    // Launch WMMA kernel
    dim3 grid(M_TILES, N_TILES);
    dim3 block(32, 1);  // One warp per block
    wmma_kernel << <grid, block >> > (device_a, device_b, device_d, M, N, K);

    // Copy result back
    cudaMemcpy(host_d.data(), device_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results
    float max_error = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        max_error = fmax(max_error, fabs(host_d[i] - host_ref[i]));
    }
    std::cout << "Max error: " << max_error << std::endl;

    // Cleanup
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_d);

    return 0;
}
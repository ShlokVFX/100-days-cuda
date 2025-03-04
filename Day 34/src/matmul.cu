#include "matmul.h"

#define TILE_SIZE 32
#define THREADS_PER_BLOCK 16

__global__ void matmul_kernel_optimized(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        size_t M, size_t N, size_t K)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int blockRow = blockIdx.y * TILE_SIZE;
    int blockCol = blockIdx.x * TILE_SIZE;

    // Each thread computes a 2x2 sub-block of C
    float sum00 = 0.0f, sum01 = 0.0f, sum10 = 0.0f, sum11 = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int tiledK = t * TILE_SIZE;

        {
            int aRow0 = blockRow + ty * 2;
            int aCol0 = tiledK + tx * 2;

            // First row of A
            if (aRow0 < M) {
                if (aCol0 + 1 < K) {
                    float2 aVal = *reinterpret_cast<const float2*>(
                        &A[aRow0 * K + aCol0]
                    );
                    As[ty * 2][tx * 2]     = aVal.x;
                    As[ty * 2][tx * 2 + 1] = aVal.y;
                } else if (aCol0 < K) {
                    As[ty * 2][tx * 2]     = A[aRow0 * K + aCol0];
                    As[ty * 2][tx * 2 + 1] = 0.0f;
                } else {
                    As[ty * 2][tx * 2]     = 0.0f;
                    As[ty * 2][tx * 2 + 1] = 0.0f;
                }
            } else {
                As[ty * 2][tx * 2]     = 0.0f;
                As[ty * 2][tx * 2 + 1] = 0.0f;
            }

            int aRow1 = aRow0 + 1;
            if (aRow1 < M) {
                if (aCol0 + 1 < K) {
                    float2 aVal = *reinterpret_cast<const float2*>(
                        &A[aRow1 * K + aCol0]
                    );
                    As[ty * 2 + 1][tx * 2]     = aVal.x;
                    As[ty * 2 + 1][tx * 2 + 1] = aVal.y;
                } else if (aCol0 < K) {
                    As[ty * 2 + 1][tx * 2]     = A[aRow1 * K + aCol0];
                    As[ty * 2 + 1][tx * 2 + 1] = 0.0f;
                } else {
                    As[ty * 2 + 1][tx * 2]     = 0.0f;
                    As[ty * 2 + 1][tx * 2 + 1] = 0.0f;
                }
            } else {
                As[ty * 2 + 1][tx * 2]     = 0.0f;
                As[ty * 2 + 1][tx * 2 + 1] = 0.0f;
            }
        }

        // --- Load B into shared memory ---
        {
            int bRow0 = tiledK + ty * 2;
            int bCol0 = blockCol + tx * 2;

            if (bRow0 < K) {
                if (bCol0 + 1 < N) {
                    float2 bVal = *reinterpret_cast<const float2*>(
                        &B[bRow0 * N + bCol0]
                    );
                    Bs[ty * 2][tx * 2]     = bVal.x;
                    Bs[ty * 2][tx * 2 + 1] = bVal.y;
                } else if (bCol0 < N) {
                    Bs[ty * 2][tx * 2]     = B[bRow0 * N + bCol0];
                    Bs[ty * 2][tx * 2 + 1] = 0.0f;
                } else {
                    Bs[ty * 2][tx * 2]     = 0.0f;
                    Bs[ty * 2][tx * 2 + 1] = 0.0f;
                }
            } else {
                Bs[ty * 2][tx * 2]     = 0.0f;
                Bs[ty * 2][tx * 2 + 1] = 0.0f;
            }

            int bRow1 = bRow0 + 1;
            if (bRow1 < K) {
                if (bCol0 + 1 < N) {
                    float2 bVal = *reinterpret_cast<const float2*>(
                        &B[bRow1 * N + bCol0]
                    );
                    Bs[ty * 2 + 1][tx * 2]     = bVal.x;
                    Bs[ty * 2 + 1][tx * 2 + 1] = bVal.y;
                } else if (bCol0 < N) {
                    Bs[ty * 2 + 1][tx * 2]     = B[bRow1 * N + bCol0];
                    Bs[ty * 2 + 1][tx * 2 + 1] = 0.0f;
                } else {
                    Bs[ty * 2 + 1][tx * 2]     = 0.0f;
                    Bs[ty * 2 + 1][tx * 2 + 1] = 0.0f;
                }
            } else {
                Bs[ty * 2 + 1][tx * 2]     = 0.0f;
                Bs[ty * 2 + 1][tx * 2 + 1] = 0.0f;
            }
        }

        __syncthreads();

        // --- Compute partial products ---
        for (int i = 0; i < TILE_SIZE; ++i) {
            float a0 = As[ty * 2][i];
            float a1 = As[ty * 2 + 1][i];
            float b0 = Bs[i][tx * 2];
            float b1 = Bs[i][tx * 2 + 1];

            sum00 += a0 * b0;
            sum01 += a0 * b1;
            sum10 += a1 * b0;
            sum11 += a1 * b1;
        }

        __syncthreads();
    }

    // --- Write results to C ---
    int cRow = blockRow + ty * 2;
    int cCol = blockCol + tx * 2;
    if (cRow < M && cCol < N) {
        C[cRow * N + cCol] = sum00;
        if (cCol + 1 < N)
            C[cRow * N + (cCol + 1)] = sum01;
        if (cRow + 1 < M) {
            C[(cRow + 1) * N + cCol] = sum10;
            if (cCol + 1 < N)
                C[(cRow + 1) * N + (cCol + 1)] = sum11;
        }
    }
}

extern "C" void solution(float* input_a, float* input_b, float* output_c,
                         size_t m, size_t n, size_t k)
{
    dim3 block(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE,
              (m + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel_optimized<<<grid, block>>>(input_a, input_b, output_c, m, n, k);
    cudaDeviceSynchronize();
}

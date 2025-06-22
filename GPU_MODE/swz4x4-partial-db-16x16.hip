#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_runtime.h>

#define BLOCK_DIM 128

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              hipGetErrorString(err));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define BENCHMARK_HIP_KERNEL(name, warmup_iters, bench_iters, kernel_call)     \
  do {                                                                         \
    printf("Benchmarking %s...\n", name);                                      \
    /* warmup */                                                               \
    for (int _i = 0; _i < warmup_iters; ++_i) {                                \
      kernel_call;                                                             \
    }                                                                          \
    HIP_CHECK(hipDeviceSynchronize());                                         \
                                                                               \
    hipEvent_t _start, _stop;                                                  \
    HIP_CHECK(hipEventCreate(&_start));                                        \
    HIP_CHECK(hipEventCreate(&_stop));                                         \
                                                                               \
    /* benchmark */                                                            \
    HIP_CHECK(hipEventRecord(_start));                                         \
    for (int _i = 0; _i < bench_iters; ++_i) {                                 \
      kernel_call;                                                             \
    }                                                                          \
    HIP_CHECK(hipEventRecord(_stop));                                          \
    HIP_CHECK(hipEventSynchronize(_stop));                                     \
                                                                               \
    float _elapsed_ms = 0.0f;                                                  \
    HIP_CHECK(hipEventElapsedTime(&_elapsed_ms, _start, _stop));               \
                                                                               \
    printf("Average runtime for %s: %.4f ms\n\n", name,                        \
           _elapsed_ms / bench_iters);                                         \
                                                                               \
    HIP_CHECK(hipEventDestroy(_start));                                        \
    HIP_CHECK(hipEventDestroy(_stop));                                         \
  } while (0)

__host__ __device__ __forceinline__ constexpr uint32_t cdiv(uint32_t a,
                                                            uint32_t b) {
  return (a + b - 1) / b;
}

__global__ void fp8_mm_ref_kernel(const __hip_fp8_e4m3_fnuz *A,
                                  const __hip_fp8_e4m3_fnuz *B,
                                  const float *A_scale, const float *B_scale,
                                  __hip_bfloat16 *C, uint32_t N, uint32_t K,
                                  uint32_t M) {
  uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= N || col >= M)
    return;

  const uint32_t SUBM = cdiv(M, BLOCK_DIM);
  const uint32_t N_BLOCKS = cdiv(K, BLOCK_DIM);

  float sum = 0.0f;
  for (uint32_t blockIdx = 0; blockIdx < N_BLOCKS; ++blockIdx) {
    float block_sum = 0.0f;
    for (uint32_t k = 0; k < BLOCK_DIM; ++k) {
      float a_fp32 = A[row + (blockIdx * BLOCK_DIM + k) * N];
      float b_fp32 = B[col + (blockIdx * BLOCK_DIM + k) * M];
      block_sum += a_fp32 * b_fp32;
    }
    // sum += block_sum;
    sum += block_sum * A_scale[row + blockIdx * N] *
           B_scale[col / BLOCK_DIM + blockIdx * SUBM];
  }

  C[row * M + col] = (__hip_bfloat16)sum;
}

template <typename T> __device__ __forceinline__ void swap(T &a, T &b) {
  T temp = a;
  a = b;
  b = temp;
}

template <const uint16_t BN, const uint16_t BK, const uint16_t BM,
          const uint16_t WITERN, const uint16_t WITERM, const uint16_t SM_COUNT,
          const uint16_t MFMA_DIM, const uint16_t MFMA_K>
__global__ void
fp8_mm_kernel_N(const __hip_fp8_e4m3_fnuz *A, const __hip_fp8_e4m3_fnuz *B,
                const float *A_scale, const float *B_scale, __hip_bfloat16 *C,
                uint16_t N, uint16_t K, uint16_t M) {
  using uint8x16 =
      __attribute__((__vector_size__(16 * sizeof(uint8_t)))) uint8_t;
  using uint8x4 = __attribute__((__vector_size__(4 * sizeof(uint8_t)))) uint8_t;
  using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

  static constexpr uint16_t VECTOR_SIZE = 4;
  static constexpr uint16_t WARPSIZE = 64;
  static constexpr uint16_t WN = MFMA_DIM * WITERN;
  static constexpr uint16_t WM = MFMA_DIM * WITERM;
  static constexpr uint16_t numThreads =
      (BN * BM) / ((MFMA_DIM * MFMA_DIM / WARPSIZE) * WITERN * WITERM);
  static constexpr uint16_t strideA =
      (numThreads / (BN / VECTOR_SIZE)) * VECTOR_SIZE;
  static constexpr uint16_t strideB =
      (numThreads / (BM / VECTOR_SIZE)) * VECTOR_SIZE;
  static constexpr uint16_t nstridesA = BK >= strideA ? cdiv(BK, strideA) : 1;
  static constexpr uint16_t nstridesB = BK >= strideB ? cdiv(BK, strideB) : 1;

  static_assert(numThreads % BN == 0, "BN should be a multiple of numThreads");
  static_assert(numThreads % BM == 0, "BM should be a multiple of numThreads");
  static_assert(BK <= 128 && BM <= 128, "Range above 128 is not supported");

  uint16_t numTiles = cdiv(N, BN) * cdiv(M, BM);

  for (uint16_t tileIdx = blockIdx.x; tileIdx < numTiles; tileIdx += SM_COUNT) {
    uint16_t rowOffsetC = (tileIdx % cdiv(N, BN)) * BN;
    uint16_t colOffsetC = (tileIdx / cdiv(N, BN)) * BM;
    uint16_t colOffsetA = rowOffsetC;
    uint16_t colOffsetB = colOffsetC;
    uint16_t M_scale = cdiv(M, BLOCK_DIM);

    uint16_t innerColA = (threadIdx.x % (BN / VECTOR_SIZE)) * VECTOR_SIZE;
    uint16_t innerRowA = (threadIdx.x / (BN / VECTOR_SIZE)) * VECTOR_SIZE;
    uint16_t innerColB = (threadIdx.x % (BM / VECTOR_SIZE)) * VECTOR_SIZE;
    uint16_t innerRowB = (threadIdx.x / (BM / VECTOR_SIZE)) * VECTOR_SIZE;

    uint16_t laneIdx = threadIdx.x % WARPSIZE;
    uint16_t warpIdx = threadIdx.x / WARPSIZE;
    uint16_t warpColOffset = (warpIdx % (BM / WM)) * WM;
    uint16_t warpRowOffset = (warpIdx / (BM / WM)) * WN;
    uint16_t warpX = laneIdx % MFMA_DIM;
    uint16_t warpY = laneIdx / MFMA_DIM;

    __shared__ __hip_fp8_e4m3_fnuz As[2][BN][BK], Bs[2][BM][BK];
    int curr = 0, next = 1;

    for (uint16_t innerRowOffsetA = 0; innerRowOffsetA < BK;
         innerRowOffsetA += strideA) {
      if ((innerRowOffsetA + innerRowA) < K && (colOffsetA + innerColA) < N &&
          (innerRowOffsetA + innerRowA) < BK) {
        uint8x4 x[VECTOR_SIZE], xt[VECTOR_SIZE];

        for (uint16_t i = 0; i < VECTOR_SIZE; ++i) {
          x[i] = *reinterpret_cast<const uint8x4 *>(
              &A[(innerRowOffsetA + innerRowA + i) * N +
                 (colOffsetA + innerColA)]);
        }

        for (uint16_t i = 0; i < VECTOR_SIZE; ++i) {
          for (uint16_t j = 0; j < VECTOR_SIZE; ++j) {
            xt[i][j] = x[j][i];
          }
        }

        for (uint16_t i = 0; i < VECTOR_SIZE; ++i) {
          int row = innerColA + i;
          int col = innerRowOffsetA + innerRowA;
col = ((col + (row >> 3) * 12) ^ ((row >> 1) & 31) + (row << 2)) & 127;
          *reinterpret_cast<uint8x4 *>(&As[curr][row][col]) = xt[i];
        }
      } else if ((innerRowOffsetA + innerRowA) < BK) {
        for (uint16_t i = 0; i < VECTOR_SIZE; ++i) {
          int row = innerColA + i;
          int col = innerRowOffsetA + innerRowA;
col = ((col + (row >> 3) * 12) ^ ((row >> 1) & 31) + (row << 2)) & 127;
          *reinterpret_cast<uint8x4 *>(&As[curr][row][col]) = {0};
        }
      }
    }
    for (uint16_t innerRowOffsetB = 0; innerRowOffsetB < BK;
         innerRowOffsetB += strideB) {
      if ((innerRowOffsetB + innerRowB) < K && (colOffsetB + innerColB) < M &&
          (innerRowOffsetB + innerRowB) < BK) {
        uint8x4 x[VECTOR_SIZE], xt[VECTOR_SIZE];

        for (uint16_t i = 0; i < VECTOR_SIZE; ++i) {
          x[i] = *reinterpret_cast<const uint8x4 *>(
              &B[(innerRowOffsetB + innerRowB + i) * M +
                 (colOffsetB + innerColB)]);
        }

        for (uint16_t i = 0; i < VECTOR_SIZE; ++i) {
          for (uint16_t j = 0; j < VECTOR_SIZE; ++j) {
            xt[i][j] = x[j][i];
          }
        }

        for (uint16_t i = 0; i < VECTOR_SIZE; ++i) {
          int row = innerColB + i;
          int col = innerRowOffsetB + innerRowB;
col = ((col + (row >> 3) * 12) ^ ((row >> 1) & 31) + (row << 2)) & 127;
          *reinterpret_cast<uint8x4 *>(&Bs[curr][row][col]) = xt[i];
        }
      } else if ((innerRowOffsetB + innerRowB) < BK) {
        for (uint16_t i = 0; i < VECTOR_SIZE; ++i) {
          int row = innerColB + i;
          int col = innerRowOffsetB + innerRowB;
col = ((col + (row >> 3) * 12) ^ ((row >> 1) & 31) + (row << 2)) & 127;
          *reinterpret_cast<uint8x4 *>(&Bs[curr][row][col]) = {0};
        }
      }
    }

    __syncthreads();

    uint32_t A_tmp[VECTOR_SIZE][nstridesA];
    uint32_t B_tmp[VECTOR_SIZE][nstridesB];

    floatx4 d[WITERN][WITERM] = {0};

    for (uint16_t tileOffset = BK; tileOffset < K + BK; tileOffset += BK) {
      if (tileOffset < K) {
        for (uint16_t innerRowOffsetA = 0; innerRowOffsetA < BK;
             innerRowOffsetA += strideA) {
          if ((tileOffset + innerRowOffsetA + innerRowA) < K &&
              (colOffsetA + innerColA) < N &&
              (innerRowOffsetA + innerRowA) < BK) {
            for (uint16_t i = 0; i < VECTOR_SIZE; ++i) {
              A_tmp[i][innerRowOffsetA / strideA] =
                  *reinterpret_cast<const uint32_t *>(
                      &A[(tileOffset + innerRowOffsetA + innerRowA + i) * N +
                         (colOffsetA + innerColA)]);
            }
          } else if ((innerRowOffsetA + innerRowA) < BK) {
            for (uint16_t i = 0; i < VECTOR_SIZE; ++i) {
              A_tmp[i][innerRowOffsetA / strideA] = {0};
            }
          }
        }
        for (uint16_t innerRowOffsetB = 0; innerRowOffsetB < BK;
             innerRowOffsetB += strideB) {
          if ((tileOffset + innerRowOffsetB + innerRowB) < K &&
              (colOffsetB + innerColB) < M &&
              (innerRowOffsetB + innerRowB) < BK) {
            for (uint16_t i = 0; i < VECTOR_SIZE; ++i) {
              B_tmp[i][innerRowOffsetB / strideB] =
                  *reinterpret_cast<const uint32_t *>(
                      &B[(tileOffset + innerRowOffsetB + innerRowB + i) * M +
                         (colOffsetB + innerColB)]);
            }
          } else if ((innerRowOffsetB + innerRowB) < BK) {
            for (uint16_t i = 0; i < VECTOR_SIZE; ++i) {
              B_tmp[i][innerRowOffsetB / strideB] = {0};
            }
          }
        }
      }

      long a[WITERN], b[WITERM];
      floatx4 c[WITERN][WITERM] = {0};
      floatx4 a_scale[WITERN];
      float b_scale = B_scale[((tileOffset - BK) / BLOCK_DIM) * M_scale +
                              (colOffsetB / BLOCK_DIM)];

      for (uint16_t wn = 0; wn < WITERN; ++wn) {
        a_scale[wn] = *reinterpret_cast<const floatx4 *>(
            &A_scale[((tileOffset - BK) / BLOCK_DIM) * N +
                     (colOffsetA + warpRowOffset + wn * MFMA_DIM + warpY * 4)]);
      }

      for (uint16_t BKOffset = 0; BKOffset < BK; BKOffset += MFMA_K) {
        for (uint16_t wn = 0; wn < WITERN; ++wn) {
          int row = warpRowOffset + wn * MFMA_DIM + warpX;
          int col = BKOffset + warpY * 8;
col = ((col + (row >> 3) * 12) ^ ((row >> 1) & 31) + (row << 2)) & 127;
          a[wn] = *reinterpret_cast<long *>(&As[curr][row][col]);
        }
        for (uint16_t wm = 0; wm < WITERM; ++wm) {
          int row = warpColOffset + wm * MFMA_DIM + warpX;
          int col = BKOffset + warpY * 8;
col = ((col + (row >> 3) * 12) ^ ((row >> 1) & 31) + (row << 2)) & 127;
          b[wm] = *reinterpret_cast<long *>(&Bs[curr][row][col]);
        }
        for (uint16_t wn = 0; wn < WITERN; ++wn) {
          for (uint16_t wm = 0; wm < WITERM; ++wm) {
            c[wn][wm] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
                a[wn], b[wm], c[wn][wm], 0, 0, 0);
          }
        }
      }
      for (uint16_t wn = 0; wn < WITERN; ++wn) {
        for (uint16_t wm = 0; wm < WITERM; ++wm) {
#pragma unroll
          for (uint16_t i = 0; i < 4; ++i) {
            d[wn][wm][i] += c[wn][wm][i] * a_scale[wn][i] * b_scale;
          }
        }
      }

      if (tileOffset < K) {
        for (uint16_t innerRowOffsetA = 0; innerRowOffsetA < BK;
             innerRowOffsetA += strideA) {
          if ((innerRowOffsetA + innerRowA) < BK) {
            uint8x4 xt[VECTOR_SIZE];
            for (uint16_t i = 0; i < VECTOR_SIZE; ++i) {
              for (uint16_t j = 0; j < VECTOR_SIZE; ++j) {
                xt[i][j] =
                    uint8_t(A_tmp[j][innerRowOffsetA / strideA] >> 8 * i);
              }
            }

            for (uint16_t i = 0; i < VECTOR_SIZE; ++i) {
              int row = innerColA + i;
              int col = innerRowOffsetA + innerRowA;
              col = (col + (row / 32) * 8) % 128;
              col = (col + row * 8) % 128;

              *reinterpret_cast<uint8x4 *>(&As[next][row][col]) = xt[i];
            }
          }
        }

        for (uint16_t innerRowOffsetB = 0; innerRowOffsetB < BK;
             innerRowOffsetB += strideB) {
          if ((innerRowOffsetB + innerRowB) < BK) {
            uint8x4 xt[VECTOR_SIZE];

            for (uint16_t i = 0; i < VECTOR_SIZE; ++i) {
              for (uint16_t j = 0; j < VECTOR_SIZE; ++j) {
                xt[i][j] =
                    uint8_t(B_tmp[j][innerRowOffsetB / strideB] >> 8 * i);
              }
            }

            for (uint16_t i = 0; i < VECTOR_SIZE; ++i) {
              int row = innerColB + i;
              int col = innerRowOffsetB + innerRowB;
              col = (col + (row / 32) * 8) % 128;
              col = (col + row * 8) % 128;
              *reinterpret_cast<uint8x4 *>(&Bs[next][row][col]) = xt[i];
            }
          }
        }
      }

      __syncthreads();

      curr = 1 - curr;
      next = 1 - next;
    }

    for (uint16_t wn = 0; wn < WITERN; ++wn) {
      for (uint16_t wm = 0; wm < WITERM; ++wm) {
#pragma unroll
        for (uint16_t i = 0; i < 4; ++i) {
          if ((rowOffsetC + warpRowOffset + wn * MFMA_DIM + warpY * 4 + i) <
                  N &&
              (colOffsetC + warpColOffset + wm * MFMA_DIM + warpX) < M)
            C[(rowOffsetC + warpRowOffset + wn * MFMA_DIM + warpY * 4 + i) * M +
              (colOffsetC + warpColOffset + wm * MFMA_DIM + warpX)] =
                (__hip_bfloat16)d[wn][wm][i];
        }
      }
    }
  }
}

bool allclose(__hip_bfloat16 *A, __hip_bfloat16 *B, uint32_t N, uint32_t M) {
  for (uint32_t i = 0; i < N; ++i) {
    for (uint32_t j = 0; j < M; ++j) {
      if (std::isnan((float)A[i * M + j]) || std::isnan((float)B[i * M + j]) ||
          abs((float)A[i * M + j] - (float)B[i * M + j]) > 2) {
        printf("Mismatch at (%lu, %lu), A = %f and B = %f\n", i, j,
               (float)A[i * M + j], (float)B[i * M + j]);
        return false;
      }
    }
  }
  return true;
}

int main() {
  uint32_t N = 6144, K = 7168, M = 576;

  __hip_fp8_e4m3_fnuz *A, *B;
  float *A_scale, *B_scale;
  __hip_bfloat16 *C, *C_base;

  __hip_fp8_e4m3_fnuz *A_d, *B_d;
  float *A_scale_d, *B_scale_d;
  __hip_bfloat16 *C_d, *C_base_d;

  A = (__hip_fp8_e4m3_fnuz *)malloc(K * N * sizeof(__hip_fp8_e4m3_fnuz));
  B = (__hip_fp8_e4m3_fnuz *)malloc(K * M * sizeof(__hip_fp8_e4m3_fnuz));
  A_scale = (float *)malloc(cdiv(K, 128) * N * sizeof(float));
  B_scale = (float *)malloc(cdiv(K, 128) * cdiv(M, 128) * sizeof(float));
  C = (__hip_bfloat16 *)malloc(N * M * sizeof(__hip_bfloat16));
  C_base = (__hip_bfloat16 *)malloc(N * M * sizeof(__hip_bfloat16));

  HIP_CHECK(hipMalloc(&A_d, K * N * sizeof(__hip_fp8_e4m3_fnuz)));
  HIP_CHECK(hipMalloc(&B_d, K * M * sizeof(__hip_fp8_e4m3_fnuz)));
  HIP_CHECK(hipMalloc(&A_scale_d, cdiv(K, 128) * N * sizeof(float)));
  HIP_CHECK(hipMalloc(&B_scale_d, cdiv(K, 128) * cdiv(M, 128) * sizeof(float)));
  HIP_CHECK(hipMalloc(&C_d, N * M * sizeof(__hip_bfloat16)));
  HIP_CHECK(hipMalloc(&C_base_d, N * M * sizeof(__hip_bfloat16)));

  for (uint32_t i = 0; i < K * N; ++i) {
    A[i] = (__hip_fp8_e4m3_fnuz)((float)rand() / RAND_MAX);
  }
  for (uint32_t i = 0; i < K * M; ++i) {
    B[i] = (__hip_fp8_e4m3_fnuz)((float)rand() / RAND_MAX);
  }
  for (uint32_t i = 0; i < cdiv(K, 128) * N; ++i) {
    A_scale[i] = (float)rand() / RAND_MAX;
  }
  for (uint32_t i = 0; i < cdiv(K, 128) * cdiv(M, 128); ++i) {
    B_scale[i] = (float)rand() / RAND_MAX;
  }

  HIP_CHECK(hipMemcpy(A_d, A, K * N * sizeof(__hip_fp8_e4m3_fnuz),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B, K * M * sizeof(__hip_fp8_e4m3_fnuz),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(A_scale_d, A_scale, cdiv(K, 128) * N * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_scale_d, B_scale,
                      cdiv(K, 128) * cdiv(M, 128) * sizeof(float),
                      hipMemcpyHostToDevice));

  {
    HIP_CHECK(hipDeviceSynchronize());
    dim3 numThreads(32, 32);
    dim3 numBlocks(cdiv(M, numThreads.x), cdiv(N, numThreads.y));
    fp8_mm_ref_kernel<<<numBlocks, numThreads>>>(A_d, B_d, A_scale_d, B_scale_d,
                                                 C_base_d, N, K, M);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
  }

  {
    const uint32_t BK = 128;
    const uint32_t BN = 128;
    const uint32_t BM = 128;
    const uint32_t WITERN = 2;
    const uint32_t WITERM = 2;
    const uint32_t SM_COUNT = 304;
    const uint32_t WARPSIZE = 64;
    const uint32_t MFMA_DIM = 16;
    const uint32_t MFMA_K = 32;
    dim3 numThreads((BN * BM) /
                    ((MFMA_DIM * MFMA_DIM / WARPSIZE) * WITERN * WITERM));
    dim3 numBlocks(SM_COUNT);
    BENCHMARK_HIP_KERNEL(
        "hip kernel", 5, 20,
        (fp8_mm_kernel_N<BN, BK, BM, WITERN, WITERM, SM_COUNT, MFMA_DIM, MFMA_K>
         <<<numBlocks, numThreads>>>(A_d, B_d, A_scale_d, B_scale_d, C_d, N, K,
                                     M)));
  }

  HIP_CHECK(
      hipMemcpy(C, C_d, N * M * sizeof(__hip_bfloat16), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(C_base, C_base_d, N * M * sizeof(__hip_bfloat16),
                      hipMemcpyDeviceToHost));

  printf("Match impl: %s\n\n", allclose(C_base, C, N, M) ? "true" : "false");

  // for (uint32_t i = 0; i < 4; ++i) {
  //   for (uint32_t j = 0; j < 4; ++j) {
  //     printf("%f\n", (float)C[i * M + j]);
  //   }
  // }

  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(A_scale_d));
  HIP_CHECK(hipFree(B_scale_d));
  HIP_CHECK(hipFree(C_d));
  HIP_CHECK(hipFree(C_base_d));

  free(A);
  free(B);
  free(A_scale);
  free(B_scale);
  free(C);
  free(C_base);

  return 0;
}
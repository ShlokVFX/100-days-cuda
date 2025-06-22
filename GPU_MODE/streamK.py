#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/amd_detail/amd_hip_fp8.h>

__host__ __device__ __forceinline__ constexpr int cdiv(int a, int b) {
  return (a + b - 1) / b;
}

#define BLOCK_DIM 128

template <const uint16_t BN, const uint16_t BK, const uint16_t BM,
          const uint16_t WITERN, const uint16_t WITERM, const uint16_t SM_COUNT,
          const uint16_t MFMA_DIM, const uint16_t MFMA_K>
__global__ void fp8_mm_kernel_streamk(const __hip_fp8_e4m3_fnuz *__restrict__ A,
                                      const __hip_fp8_e4m3_fnuz *__restrict__ B,
                                      const float *__restrict__ A_scale,
                                      const float *__restrict__ B_scale, float *__restrict__ C,
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

  uint32_t numTiles = cdiv(N, BN) * cdiv(M, BM);
  uint16_t subTilesPerTile = cdiv(K, BK);
  uint32_t totalSubTiles = numTiles * subTilesPerTile;
  uint16_t subTilesToProcess = cdiv(totalSubTiles, SM_COUNT);
  uint32_t startSubTileIdx = subTilesToProcess * blockIdx.x;
  uint32_t startTileIdx = startSubTileIdx / subTilesPerTile;
  uint32_t subTile = startSubTileIdx;

  for (uint16_t tileIdx = startTileIdx, subTilesProcessed = 0;
       tileIdx < numTiles && subTilesProcessed < subTilesToProcess; ++tileIdx) {
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

    subTile = subTile % subTilesPerTile;
    uint16_t tileOffset = subTile * BK;

    for (uint16_t innerRowOffsetA = 0; innerRowOffsetA < BK;
         innerRowOffsetA += strideA) {
      if ((tileOffset + innerRowOffsetA + innerRowA) < K &&
          (colOffsetA + innerColA) < N && (innerRowOffsetA + innerRowA) < BK) {
        uint8x4 x[VECTOR_SIZE], xt[VECTOR_SIZE];

        for (uint16_t i = 0; i < VECTOR_SIZE; ++i) {
          x[i] = *reinterpret_cast<const uint8x4 *>(
              &A[(tileOffset + innerRowOffsetA + innerRowA + i) * N +
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

          col = (col + (row / 32) * 8) % 128;
          col = (col + row * 8) % 128;
          
          *reinterpret_cast<uint8x4 *>(&As[curr][row][col]) = xt[i];
        }
      } else if ((innerRowOffsetA + innerRowA) < BK) {
        for (uint16_t i = 0; i < VECTOR_SIZE; ++i) {
          int row = innerColA + i;
          int col = innerRowOffsetA + innerRowA;
          col = (col + (row / 32) * 8) % 128;
          col = (col + row * 8) % 128;
          *reinterpret_cast<uint8x4 *>(&As[curr][row][col]) = {0};
        }
      }
    }
    for (uint16_t innerRowOffsetB = 0; innerRowOffsetB < BK;
         innerRowOffsetB += strideB) {
      if ((tileOffset + innerRowOffsetB + innerRowB) < K &&
          (colOffsetB + innerColB) < M && (innerRowOffsetB + innerRowB) < BK) {
        uint8x4 x[VECTOR_SIZE], xt[VECTOR_SIZE];

        for (uint16_t i = 0; i < VECTOR_SIZE; ++i) {
          x[i] = *reinterpret_cast<const uint8x4 *>(
              &B[(tileOffset + innerRowOffsetB + innerRowB + i) * M +
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
          col = (col + (row / 32) * 8) % 128;
          col = (col + row * 8) % 128;
          *reinterpret_cast<uint8x4 *>(&Bs[curr][row][col]) = xt[i];
        }
      } else if ((innerRowOffsetB + innerRowB) < BK) {
        for (uint16_t i = 0; i < VECTOR_SIZE; ++i) {
          int row = innerColB + i;
          int col = innerRowOffsetB + innerRowB;
          col = (col + (row / 32) * 8) % 128;
          col = (col + row * 8) % 128;
          *reinterpret_cast<uint8x4 *>(&Bs[curr][row][col]) = {0};
        }
      }
    }

    __syncthreads();

    uint32_t A_tmp[VECTOR_SIZE][nstridesA];
    uint32_t B_tmp[VECTOR_SIZE][nstridesB];

    floatx4 d[WITERN][WITERM] = {0};

    for (tileOffset += BK;
         subTile < subTilesPerTile && subTilesProcessed < subTilesToProcess;
         ++subTile, ++subTilesProcessed, tileOffset += BK) {
      if (tileOffset < K && subTilesProcessed + 1 < subTilesToProcess) {
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
      float b_scale = B_scale[((tileOffset - BK) / BLOCK_DIM) * M_scale +
                              (colOffsetB / BLOCK_DIM)];

      for (uint16_t BKOffset = 0; BKOffset < BK; BKOffset += MFMA_K) {
        for (uint16_t wn = 0; wn < WITERN; ++wn) {
          int row = warpRowOffset + wn * MFMA_DIM + warpX;
          int col = BKOffset + warpY * 8;
          col = (col + (row / 32) * 8) % 128;
          col = (col + row * 8) % 128;
          a[wn] = *reinterpret_cast<const uint64_t *>(&As[curr][row][col]);
        }
        for (uint16_t wm = 0; wm < WITERM; ++wm) {
          int row = warpColOffset + wm * MFMA_DIM + warpX;
          int col = BKOffset + warpY * 8;
          col = (col + (row / 32) * 8) % 128;
          col = (col + row * 8) % 128;
          b[wm] = *reinterpret_cast<const uint64_t *>(&Bs[curr][row][col]);
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
          floatx4 a_scale = *reinterpret_cast<const floatx4 *>(
              &A_scale[((tileOffset - BK) / BLOCK_DIM) * N +
                       (colOffsetA + warpRowOffset + wn * MFMA_DIM +
                        warpY * 4)]);
#pragma unroll
          for (uint16_t i = 0; i < 4; ++i) {
            d[wn][wm][i] += c[wn][wm][i] * a_scale[i] * b_scale;
          }
        }
      }

      if (tileOffset < K && subTilesProcessed + 1 < subTilesToProcess) {
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

            for (uint16_t i = 0; i < 4; ++i) {
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
        for (uint16_t i = 0; i < 4; ++i) {
          if ((rowOffsetC + warpRowOffset + wn * MFMA_DIM + warpY * 4 + i) <
                  N &&
              (colOffsetC + warpColOffset + wm * MFMA_DIM + warpX) < M) {
            atomicAdd(&C[(rowOffsetC + warpRowOffset + wn * MFMA_DIM +
                          warpY * 4 + i) *
                             M +
                         (colOffsetC + warpColOffset + wm * MFMA_DIM + warpX)],
                      d[wn][wm][i]);
          }
        }
      }
    }
  }
}

  at::Tensor fp8_mm(at::Tensor A, at::Tensor B, at::Tensor A_scale,
                  at::Tensor B_scale, at::Tensor C) {
  int N = A.size(0), K = A.size(1), M = B.size(0);

  static float *C_dataptr = NULL;
  if (!C_dataptr) {
    hipMalloc(&C_dataptr, 6144 * 7168 * sizeof(float));
  }

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

  if (K >= 7168 && N <= 1024 && M <= 1536) {
    hipMemset(C_dataptr, 0, N * M * sizeof(float));
    at::Tensor Cf32 = at::from_blob(
        C_dataptr, {N, M},
        at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));

    fp8_mm_kernel_streamk<BN, BK, BM, WITERN, WITERM, SM_COUNT, MFMA_DIM,
                          MFMA_K><<<numBlocks, numThreads>>>(
        (__hip_fp8_e4m3_fnuz *)A.data_ptr(),
        (__hip_fp8_e4m3_fnuz *)B.data_ptr(), A_scale.data_ptr<float>(),
        B_scale.data_ptr<float>(), (float *)Cf32.data_ptr(), N, K, M);
    return Cf32;
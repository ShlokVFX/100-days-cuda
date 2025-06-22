import torch
import time
import numpy as np
import os
import sys
from torch.utils.cpp_extension import load_inline



os.environ["CXX"] = "clang++"

cuda_src = r"""
#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/amd_detail/amd_hip_fp8.h>

__host__ __device__ __forceinline__ constexpr int cdiv(int a, int b) {
  return (a + b - 1) / b;
}

#define float16_t __fp16

template <const uint32_t BN, const uint32_t BK, const uint32_t BM,
          const uint32_t WITERN, const uint32_t WITERM, const uint32_t SM_COUNT,
          bool COL_SCHEDULING>
__global__ void fp16_mm_kernel(const float16_t *A, const float16_t *B,
                               float16_t *C, uint32_t N, uint32_t K,
                               uint32_t M) {
  using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;
  using float16x4 =
      __attribute__((__vector_size__(4 * sizeof(float16_t)))) float16_t;

  static constexpr uint32_t VECTOR_SIZE = 4;
  static constexpr uint32_t WARPSIZE = 64;
  static constexpr uint32_t WN = 32 * WITERN;
  static constexpr uint32_t WM = 32 * WITERM;
  static constexpr uint32_t numThreads = (BN * BM) / (16 * WITERN * WITERM);
  static constexpr uint32_t strideA = numThreads / (BK / VECTOR_SIZE);
  static constexpr uint32_t strideB = (numThreads / BM) * VECTOR_SIZE;
  static constexpr uint32_t nstridesA = BN > strideA ? BN / strideA : 1;
  static constexpr uint32_t nstridesB = BK > strideA ? BK / strideB : 1;

  static_assert(numThreads % (BK / VECTOR_SIZE) == 0,
                "BK / VECTOR_SIZE should be a multiple of numThreads");
  static_assert(numThreads % BM == 0, "BM should be a multiple of numThreads");

  uint32_t numTiles = cdiv(N, BN) * cdiv(M, BM);

  for (uint32_t tileIdx = blockIdx.x; tileIdx < numTiles; tileIdx += SM_COUNT) {
    uint32_t rowOffsetC, colOffsetC;
    if constexpr (COL_SCHEDULING) {
      rowOffsetC = (tileIdx % cdiv(N, BN)) * BN;
      colOffsetC = (tileIdx / cdiv(N, BN)) * BM;
    } else {
      rowOffsetC = (tileIdx / cdiv(M, BM)) * BN;
      colOffsetC = (tileIdx % cdiv(M, BM)) * BM;
    }
    uint32_t rowOffsetA = rowOffsetC;
    uint32_t colOffsetB = colOffsetC;

    uint32_t innerColA = (threadIdx.x % (BK / VECTOR_SIZE)) * VECTOR_SIZE;
    uint32_t innerRowA = threadIdx.x / (BK / VECTOR_SIZE);
    uint32_t innerColB = threadIdx.x % BM;
    uint32_t innerRowB = (threadIdx.x / BM) * VECTOR_SIZE;

    uint32_t laneIdx = threadIdx.x % WARPSIZE;
    uint32_t warpIdx = threadIdx.x / WARPSIZE;
    uint32_t warpColOffset = (warpIdx % (BM / WM)) * WM;
    uint32_t warpRowOffset = (warpIdx / (BM / WM)) * WN;
    uint32_t warpX = laneIdx % 32;
    uint32_t warpY = laneIdx / 32;

    __shared__ float16_t As[2][BN][BK], Bs[2][BM][BK + 2];
    uint32_t curr = 0;

    for (uint32_t innerRowOffsetA = 0; innerRowOffsetA < BN;
         innerRowOffsetA += strideA) {
      if ((rowOffsetA + innerRowOffsetA + innerRowA) < N && (innerColA) < K &&
          (innerRowOffsetA + innerRowA) < BN) {
        *reinterpret_cast<double *>(
            &As[curr][innerRowOffsetA + innerRowA][innerColA]) =
            *reinterpret_cast<const double *>(
                &A[(rowOffsetA + innerRowOffsetA + innerRowA) * K +
                   (innerColA)]);
      } else if ((innerRowOffsetA + innerRowA) < BN) {
        *reinterpret_cast<double *>(
            &As[curr][innerRowOffsetA + innerRowA][innerColA]) = 0.0;
      }
    }
    for (uint32_t innerRowOffsetB = 0; innerRowOffsetB < BK;
         innerRowOffsetB += strideB) {
      if ((innerRowOffsetB + innerRowB) < K && (colOffsetB + innerColB) < M &&
          (innerRowOffsetB + innerRowB) < BK) {
        uint64_t x = *reinterpret_cast<const uint16_t *>(
            &B[(innerRowOffsetB + innerRowB) * M + (colOffsetB + innerColB)]);
        uint64_t y = *reinterpret_cast<const uint16_t *>(
            &B[(innerRowOffsetB + innerRowB + 1) * M +
               (colOffsetB + innerColB)]);
        uint64_t z = *reinterpret_cast<const uint16_t *>(
            &B[(innerRowOffsetB + innerRowB + 2) * M +
               (colOffsetB + innerColB)]);
        uint64_t w = *reinterpret_cast<const uint16_t *>(
            &B[(innerRowOffsetB + innerRowB + 3) * M +
               (colOffsetB + innerColB)]);

        uint64_t pack = (w & 0xffff) << 48 | (z & 0xffff) << 32 |
                        (y & 0xffff) << 16 | (x & 0xffff);

        *reinterpret_cast<uint64_t *>(
            &Bs[curr][innerColB][innerRowOffsetB + innerRowB]) = pack;
      } else if ((innerRowOffsetB + innerRowB) < BK) {
        *reinterpret_cast<uint64_t *>(
            &Bs[curr][innerColB][innerRowOffsetB + innerRowB]) = 0;
      }
    }

    __syncthreads();

    floatx16 d[WITERN][WITERM] = {0};
    double A_tmp[nstridesA];
    uint16_t B_tmp1[nstridesB], B_tmp2[nstridesB], B_tmp3[nstridesB],
        B_tmp4[nstridesB];

    for (uint32_t tileOffset = BK; tileOffset < K + BK; tileOffset += BK) {
      // load from global to shared memory in coalesced manner
      if (tileOffset < K) {
        for (uint32_t innerRowOffsetA = 0; innerRowOffsetA < BN;
             innerRowOffsetA += strideA) {
          if ((rowOffsetA + innerRowOffsetA + innerRowA) < N &&
              (tileOffset + innerColA) < K &&
              (innerRowOffsetA + innerRowA) < BN) {
            A_tmp[innerRowOffsetA / strideA] =
                *reinterpret_cast<const double *>(
                    &A[(rowOffsetA + innerRowOffsetA + innerRowA) * K +
                       (tileOffset + innerColA)]);
          } else if ((innerRowOffsetA + innerRowA) < BN) {
            A_tmp[innerRowOffsetA / strideA] = 0.0;
          }
        }
        for (uint32_t innerRowOffsetB = 0; innerRowOffsetB < BK;
             innerRowOffsetB += strideB) {
          if ((tileOffset + innerRowOffsetB + innerRowB) < K &&
              (colOffsetB + innerColB) < M &&
              (innerRowOffsetB + innerRowB) < BK) {
            B_tmp1[innerRowOffsetB / strideB] =
                *reinterpret_cast<const uint16_t *>(
                    &B[(tileOffset + innerRowOffsetB + innerRowB) * M +
                       (colOffsetB + innerColB)]);
            B_tmp2[innerRowOffsetB / strideB] =
                *reinterpret_cast<const uint16_t *>(
                    &B[(tileOffset + innerRowOffsetB + innerRowB + 1) * M +
                       (colOffsetB + innerColB)]);
            B_tmp3[innerRowOffsetB / strideB] =
                *reinterpret_cast<const uint16_t *>(
                    &B[(tileOffset + innerRowOffsetB + innerRowB + 2) * M +
                       (colOffsetB + innerColB)]);
            B_tmp4[innerRowOffsetB / strideB] =
                *reinterpret_cast<const uint16_t *>(
                    &B[(tileOffset + innerRowOffsetB + innerRowB + 3) * M +
                       (colOffsetB + innerColB)]);
          } else if ((innerRowOffsetB + innerRowB) < BK) {
            B_tmp1[innerRowOffsetB / strideB] = 0;
            B_tmp2[innerRowOffsetB / strideB] = 0;
            B_tmp3[innerRowOffsetB / strideB] = 0;
            B_tmp4[innerRowOffsetB / strideB] = 0;
          }
        }
      }

      float16x4 a[WITERN], b[WITERM];
      for (uint32_t BKOffset = 0; BKOffset < BK; BKOffset += 8) {
        for (uint32_t wn = 0; wn < WITERN; ++wn) {
          a[wn] = *reinterpret_cast<float16x4 *>(
              &As[curr][warpRowOffset + wn * 32 + warpX][BKOffset + warpY * 4]);
        }
        for (uint32_t wm = 0; wm < WITERM; ++wm) {
          b[wm] = *reinterpret_cast<float16x4 *>(
              &Bs[curr][warpColOffset + wm * 32 + warpX][BKOffset + warpY * 4]);
        }
        for (uint32_t wn = 0; wn < WITERN; ++wn) {
          for (uint32_t wm = 0; wm < WITERM; ++wm) {
            d[wn][wm] = __builtin_amdgcn_mfma_f32_32x32x8f16(
                a[wn], b[wm], d[wn][wm], 0, 0, 0);
          }
        }
      }

      if (tileOffset < K) {
        for (uint32_t innerRowOffsetA = 0; innerRowOffsetA < BN;
             innerRowOffsetA += strideA) {
          if ((innerRowOffsetA + innerRowA) < BN) {
            *reinterpret_cast<double *>(
                &As[1 - curr][innerRowOffsetA + innerRowA][innerColA]) =
                A_tmp[innerRowOffsetA / strideA];
          }
        }
        for (uint32_t innerRowOffsetB = 0; innerRowOffsetB < BK;
             innerRowOffsetB += strideB) {
          if ((innerRowOffsetB + innerRowB) < BK) {
            uint64_t pack =
                ((uint64_t)B_tmp4[innerRowOffsetB / strideB] & 0xffff) << 48 |
                ((uint64_t)B_tmp3[innerRowOffsetB / strideB] & 0xffff) << 32 |
                ((uint64_t)B_tmp2[innerRowOffsetB / strideB] & 0xffff) << 16 |
                ((uint64_t)B_tmp1[innerRowOffsetB / strideB] & 0xffff);

            *reinterpret_cast<uint64_t *>(
                &Bs[1 - curr][innerColB][innerRowOffsetB + innerRowB]) = pack;
          }
        }
      }

      __syncthreads();

      curr = 1 - curr;
    }

    for (uint32_t wn = 0; wn < WITERN; ++wn) {
      for (uint32_t wm = 0; wm < WITERM; ++wm) {
        for (uint32_t j = 0; j < 4; ++j) {
          for (uint32_t i = 0; i < 4; ++i) {
            if ((rowOffsetC + warpRowOffset + wn * 32 + j * 8 + warpY * 4 + i) <
                    N &&
                (colOffsetC + warpColOffset + wm * 32 + warpX) < M)
              C[(rowOffsetC + warpRowOffset + wn * 32 + j * 8 + warpY * 4 + i) *
                    M +
                (colOffsetC + warpColOffset + wm * 32 + warpX)] =
                  d[wn][wm][i + j * 4];
          }
        }
      }
    }
  }
}

at::Tensor fp16_mm(at::Tensor A, at::Tensor B) {
  int N, K, M;
  if (A.sizes().size() == 3) {
    N = A.size(1);
    K = A.size(2);
  } else {
    N = A.size(0);
    K = A.size(1);
  }
  M = B.size(1);
  at::Tensor C = at::empty({N, M}, A.options());

  const uint32_t BK = 16;
  const uint32_t BN = 128;
  const uint32_t BM = 128;
  const uint32_t WITERM = 1;
  const uint32_t WITERN = 1;
  const uint32_t SM_COUNT = 304;
  const bool COL_SCHEDULING = cdiv(N, BN) > cdiv(M, BM);
  dim3 numThreads((BN * BM) / (16 * WITERN * WITERM));
  dim3 numBlocks(SM_COUNT);
  if (COL_SCHEDULING) {
    fp16_mm_kernel<BN, BK, BM, WITERN, WITERM, SM_COUNT, true>
       <<<numBlocks, numThreads>>>((float16_t *)A.data_ptr(), (float16_t *)B.data_ptr(),
                                   (float16_t *)C.data_ptr(), N, K, M);
  } else {
    fp16_mm_kernel<BN, BK, BM, WITERN, WITERM, SM_COUNT, false>
       <<<numBlocks, numThreads>>>((float16_t *)A.data_ptr(), (float16_t *)B.data_ptr(),
                                   (float16_t *)C.data_ptr(), N, K, M);
  }
  hipDeviceSynchronize();
  return C;
}
"""

cpp_src = r"""
at::Tensor fp16_mm(at::Tensor A, at::Tensor B);
"""

if sys.stdout is None:
    sys.stdout = open("/dev/stdout", "w")
if sys.stderr is None:
    sys.stderr = open("/dev/stderr", "w")

module = load_inline(
    name="fp16_mm",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["fp16_mm"],
    extra_cuda_cflags=[
        "-O3",
        "--offload-arch=gfx942",
        "-std=c++20",
    ],
    verbose=True,
)

N, K, M = 8192, 7168, 2048
a = torch.randn(N, K, dtype=torch.float16, device='cuda')
b = torch.randn(K, M, dtype=torch.float16, device='cuda')

y = a @ b

NUM_WARMUP = 5
NUM_RUNS = 20

for i in range(NUM_WARMUP):
    # pred = module.fp16_mm(a, b)
    pred = a @ b
    torch.cuda.synchronize()

times = []
for i in range(NUM_RUNS):
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    # pred = module.fp16_mm(a, b)
    pred = a @ b
    torch.cuda.synchronize()
    times.append(time.perf_counter_ns() - start)
    # if not torch.allclose(y, pred, atol=1e-6):
    #     print(y)
    #     print(pred)
    #     print((y - pred).abs().max())
    #     raise "Mismatch error"

times = np.array(times)
mean = times.mean()

if mean / 1_000_000_000 > 1:
    factor = 1_000_000_000
    suffix = "s"
elif mean / 1_000_000 > 1:
    factor = 1_000_000
    suffix = "ms"
elif mean / 1_000 > 1:
    factor = 1_000
    suffix = "µs"
else:
    factor = 1
    suffix = "µs"

times = times / factor

print("\n=============================================")
print(f"⏱ {times.mean():.2f} ± {times.std():.3f} {suffix}")
print(
    f"⚡ {times.mean() - times.std():.2f} {suffix} 🐌 {times.mean() + times.std():.2f} {suffix}"
)
# print("Match:", torch.allclose(y, pred, atol=1e-6))
print("=============================================")
print(pred)
print(y)
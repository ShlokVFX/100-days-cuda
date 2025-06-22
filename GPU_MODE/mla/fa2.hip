#include <cmath>
#include <cstdint>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/amd_detail/amd_hip_runtime.h>

constexpr uint32_t cdiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

typedef __hip_bfloat16 bf16_t;

template <const uint32_t Bc, const uint32_t numThreads, const uint32_t dqk,
          const uint32_t dv>
__global__ void
decoding_flash_attention2_kernel(bf16_t *Q, bf16_t *K, bf16_t *V, bf16_t *O,
                                 const uint32_t B, const uint32_t nh,
                                 uint32_t N) {
  using uint16x2 =
      __attribute__((__vector_size__(2 * sizeof(uint16_t)))) uint16_t;
  using uint16x4 =
      __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
  using uint16x8 =
      __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
  using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

  static constexpr uint32_t VECTOR_SIZE_K = 8;
  static constexpr uint32_t VECTOR_SIZE_V = 4;
  static constexpr uint32_t WARPSIZE = 64;
  static constexpr uint32_t N_WARPS = numThreads / WARPSIZE;
  static constexpr uint32_t rowStrideV =
      (numThreads / (dv / VECTOR_SIZE_V)) * VECTOR_SIZE_V;

  static_assert(numThreads % (dv / VECTOR_SIZE_V) == 0,
                "numThreads should be divisible by dv / VECTOR_SIZE_V");
  static_assert(numThreads == Bc, "numThreads should be equal to Bc");

  uint64_t batchOffsetQ = (uint64_t)blockIdx.x * dqk;
  uint64_t batchOffsetK = (uint64_t)blockIdx.x * N * dqk;
  uint64_t batchOffsetV = (uint64_t)blockIdx.x * N * dv;
  uint64_t batchOffsetO = (uint64_t)blockIdx.x * dv;
  uint32_t tid = threadIdx.x;
  uint32_t colV = (tid % (dv / VECTOR_SIZE_V)) * VECTOR_SIZE_V;
  uint32_t rowV = (tid / (dv / VECTOR_SIZE_V)) * VECTOR_SIZE_V;

  uint32_t laneIdx = tid % WARPSIZE;
  uint32_t warpIdx = tid / WARPSIZE;

  __shared__ bf16_t Qs[dqk];
  __shared__ float Ss[Bc];
  __shared__ float Os[dv];

  float m = -INFINITY, l = 1.0f;

  Ss[tid] = 0.0f;
  // load data from global to smem
  for (uint32_t dqkIdx = tid; dqkIdx < dqk; dqkIdx += numThreads) {
    Qs[dqkIdx] = Q[batchOffsetQ + dqkIdx];
  }

  __syncthreads();

  for (uint32_t tileOffset = 0; tileOffset < N - 1; tileOffset += Bc) {
    for (uint32_t KsIdx = tid * VECTOR_SIZE_K; KsIdx < Bc * dqk;
         KsIdx += numThreads * VECTOR_SIZE_K) {
      uint32_t row = KsIdx / dqk;
      uint32_t col = KsIdx % dqk;

      uint16x8 qpack = *reinterpret_cast<uint16x8 *>(&Qs[col]);
      uint16x8 kpack = *reinterpret_cast<uint16x8 *>(
          &K[batchOffsetK + tileOffset * dqk + KsIdx]);

      float sum = 0.0f;
      for (uint32_t i = 0; i < VECTOR_SIZE_K; ++i) {
        uint16_t qu = qpack[i];
        uint16_t ku = kpack[i];
        float q = float(*reinterpret_cast<bf16_t *>(&qu));
        float k = float(*reinterpret_cast<bf16_t *>(&ku));
        sum += q * k;
      }
      atomicAdd(&Ss[row], sum); // TODO: use DPP
    }

    __syncthreads();

    float val = (float)bf16_t(Ss[tid]) / sqrtf(dqk);
    float m_local = val, l_local = 1.0f;

    for (uint32_t s = WARPSIZE / 2; s > 0; s /= 2) {
      float m_other = __shfl_down(m_local, s);
      float l_other = __shfl_down(l_local, s);
      if (m_other > m_local) {
        l_local *= expf(m_local - m_other);
        m_local = m_other;
      }
      l_local += l_other * expf(m_other - m_local);
    }

    m_local = __shfl(m_local, 0);
    l_local = __shfl(l_local, 0);

    float m_prev = m;

    if (m_local > m) {
      l *= expf(m - m_local);
      m = m_local;
    }
    l += l_local * expf(m_local - m);

    Ss[tid] = expf(val - m);

    for (uint32_t dvIdx = tid * 2; dvIdx < dv; dvIdx += numThreads * 2) {
      *reinterpret_cast<float2 *>(&Os[dvIdx]) =
          *reinterpret_cast<float2 *>(&Os[dvIdx]) * expf(m_prev - m);
    }

    __syncthreads();

    for (uint32_t rowOffsetV = 0; rowOffsetV < Bc; rowOffsetV += rowStrideV) {
      uint16x4 x[VECTOR_SIZE_V], xt[VECTOR_SIZE_V];
      floatx4 y;

      for (uint32_t i = 0; i < VECTOR_SIZE_V; ++i) {
        x[i] = *reinterpret_cast<uint16x4 *>(
            &V[batchOffsetV + (tileOffset + rowOffsetV + rowV + i) * dv +
               colV]);
      }

      for (uint32_t i = 0; i < VECTOR_SIZE_V; ++i) {
        for (uint32_t j = 0; j < VECTOR_SIZE_V; ++j) {
          xt[i][j] = x[j][i];
        }
      }

      y = *reinterpret_cast<floatx4 *>(&Ss[rowOffsetV + rowV]);

      for (uint32_t i = 0; i < VECTOR_SIZE_V; ++i) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < VECTOR_SIZE_V; ++j) {
          uint16_t au = xt[i][j];
          sum += float(*reinterpret_cast<bf16_t *>(&au)) * y[j];
        }
        atomicAdd(&Os[colV + i], sum);
      }
    }

    Ss[tid] = 0.0f;
    __syncthreads();
  }

  float sum = 0.0f;
  for (uint32_t dqkIdx = tid; dqkIdx < dqk; dqkIdx += numThreads) {
    sum += float(Qs[dqkIdx]) * float(K[batchOffsetK + (N - 1) * dqk + dqkIdx]);
  }
  for (uint32_t s = WARPSIZE / 2; s > 0; s /= 2) {
    sum += __shfl_down(sum, s);
  }
  sum = (float)bf16_t(__shfl(sum, 0)) / sqrtf(dqk);

  float m_prev = m;

  if (sum > m) {
    l *= expf(m - sum);
    m = sum;
  }
  l += expf(sum - m);

  for (uint32_t dvIdx = tid * 2; dvIdx < dv; dvIdx += numThreads * 2) {
    __hip_bfloat162 last = *reinterpret_cast<__hip_bfloat162 *>(
        &V[batchOffsetV + (N - 1) * dv + dvIdx]);

    float2 packf = *reinterpret_cast<float2 *>(&Os[dvIdx]);

    last.x = (packf.x * expf(m_prev - m) + (float)last.x * expf(sum - m)) / l;
    last.y = (packf.y * expf(m_prev - m) + (float)last.y * expf(sum - m)) / l;
    *reinterpret_cast<__hip_bfloat162 *>(&O[batchOffsetO + dvIdx]) = last;
  }
}

int main() {
  const uint32_t B = 128;
  const uint32_t nh = 128;
  const uint32_t N = 6145;
  const uint32_t Bc = 64;
  const uint32_t Tc = cdiv(N, Bc);
  const uint32_t dqk = 192;
  const uint32_t dv = 128;
  const uint32_t numThreads = Bc;
  dim3 numBlocks(B * nh);
  decoding_flash_attention2_kernel<Bc, numThreads, dqk, dv>
      <<<numBlocks, numThreads>>>(nullptr, nullptr, nullptr, nullptr, B, nh, N);
  return 0;
  // hipDeviceSynchronize();

  // size_t B, nh, N, d, Br, Bc;
  // float *query, *key, *value, *output, *sums, *maxes;

  // B = 16, nh = 16, N = 128, d = 64;
  // Br = 24, Bc = 24;

  // query = (float *)malloc((B * nh * N * d) * sizeof(float));
  // key = (float *)malloc((B * nh * N * d) * sizeof(float));
  // value = (float *)malloc((B * nh * N * d) * sizeof(float));
  // output = (float *)malloc((B * nh * N * d) * sizeof(float));
  // sums = (float *)malloc((B * nh * N) * sizeof(float));
  // maxes = (float *)malloc((B * nh * N) * sizeof(float));

  // for (size_t i = 0; i < B * nh * N * d; ++i) {
  //   query[i] = (float)rand() / RAND_MAX;
  //   key[i] = (float)rand() / RAND_MAX;
  //   value[i] = (float)rand() / RAND_MAX;
  //   output[i] = 0;
  // }

  // for (size_t i = 0; i < B * nh * N; ++i) {
  //   sums[i] = 0;
  //   maxes[i] = -INFINITY;
  // }

  // flash_attention_gpu(query, key, value, output, sums, maxes, B, nh, N, d,
  // Br,
  //                     Bc);

  // // for (size_t i = 0; i < B * nh * N; ++i) {
  // //   for (size_t j = 0; j < d; ++j) {
  // //     printf(" %.4f", output[i * d + j]);
  // //   }
  // //   printf("\n");
  // // }

  // free(query);
  // free(key);
  // free(value);
  // free(output);
  // free(sums);
  // free(maxes);
}
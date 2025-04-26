#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <chrono>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/amd_detail/amd_hip_fp8.h>

// Helper function for ceiling division
__host__ __device__ __forceinline__ size_t cdiv(size_t a, size_t b) {
  return (a + b - 1) / b;
}

#define BLOCK_DIM 128

// Benchmark configuration struct
struct BenchmarkConfig {
    size_t k;
    size_t m;
    size_t n;
    int seed;
};

template <const size_t BN, const size_t BK, const size_t BM,
          const size_t WITERN, const size_t WITERM>
__global__ void fp8_mm_kernel(const __hip_fp8_e4m3_fnuz *A,
                              const __hip_fp8_e4m3_fnuz *B,
                              const float *A_scale, const float *B_scale,
                              __hip_bfloat16 *C, size_t N, size_t K, size_t M) {
  using floatx16 = __attribute__((__vector_size__(16 * sizeof(float)))) float;

  static constexpr size_t VECTOR_SIZE = 4;
  static constexpr size_t WARPSIZE = 64;
  static constexpr size_t WN = 32 * WITERN;
  static constexpr size_t WM = 32 * WITERM;
  static constexpr size_t numThreads = (BN * BM) / (16 * WITERN * WITERM);
  static constexpr size_t SUBBN = BN / VECTOR_SIZE;
  static constexpr size_t SUBBM = BM / VECTOR_SIZE;
  static constexpr size_t strideA = numThreads / SUBBN;
  static constexpr size_t strideB = numThreads / SUBBM;

  size_t rowOffsetC = blockIdx.y * BN;
  size_t colOffsetC = blockIdx.x * BM;
  size_t colOffsetA = rowOffsetC;
  size_t colOffsetB = colOffsetC;
  size_t M_scale = cdiv(M, BLOCK_DIM);

  static_assert(numThreads % BN == 0, "BN should be a multiple of numThreads");
  static_assert(numThreads % BM == 0, "BM should be a multiple of numThreads");
  static_assert(BK <= 128 && BM <= 128, "Range above 128 is not supported");

  size_t innerColA = threadIdx.x % SUBBN;
  size_t innerRowA = threadIdx.x / SUBBN;
  size_t innerColB = threadIdx.x % SUBBM;
  size_t innerRowB = threadIdx.x / SUBBM;

  size_t laneIdx = threadIdx.x % WARPSIZE;
  size_t warpIdx = threadIdx.x / WARPSIZE;
  size_t warpColOffset = (warpIdx % (BM / WM)) * WM;
  size_t warpRowOffset = (warpIdx / (BM / WM)) * WN;
  size_t warpX = laneIdx % 32;
  size_t warpY = laneIdx / 32;

  __shared__ __hip_fp8_e4m3_fnuz As[BK][BN], Bs[BK][BM];
  __shared__ float Ws[BN + 1];

  __hip_fp8_e4m3_fnuz a[WITERN][8], b[WITERM][8];
  floatx16 d[WITERN][WITERM] = {0};
  float b_scale;

  for (size_t tileOffset = 0; tileOffset < K; tileOffset += BK) {
    // load from global to shared memory in coalesced manner
    for (size_t innerRowOffsetA = 0; innerRowOffsetA < BK;
         innerRowOffsetA += strideA) {
      if ((tileOffset + innerRowOffsetA + innerRowA) < K &&
          (colOffsetA + innerColA * VECTOR_SIZE) < N &&
          (innerRowOffsetA + innerRowA) < BK) {
        *reinterpret_cast<float *>(
            &As[innerRowOffsetA + innerRowA][innerColA * VECTOR_SIZE]) =
            *reinterpret_cast<const float *>(
                &A[(tileOffset + innerRowOffsetA + innerRowA) * N +
                   (colOffsetA + innerColA * VECTOR_SIZE)]);
      } else if ((innerRowOffsetA + innerRowA) < BK) {
        *reinterpret_cast<float *>(
            &As[innerRowOffsetA + innerRowA][innerColA * VECTOR_SIZE]) = 0.0f;
      }
    }
    if (threadIdx.x < BN) {
      *reinterpret_cast<float4 *>(&Ws[threadIdx.x * VECTOR_SIZE]) =
          *reinterpret_cast<const float4 *>(
              &A_scale[(tileOffset / BLOCK_DIM) * N +
                       (colOffsetA + threadIdx.x * VECTOR_SIZE)]);
    }
    for (size_t innerRowOffsetB = 0; innerRowOffsetB < BK;
         innerRowOffsetB += strideB) {
      if ((tileOffset + innerRowOffsetB + innerRowB) < K &&
          (colOffsetB + innerColB * VECTOR_SIZE) < M &&
          (innerRowOffsetB + innerRowB) < BK) {
        *reinterpret_cast<float *>(
            &Bs[innerRowOffsetB + innerRowB][innerColB * VECTOR_SIZE]) =
            *reinterpret_cast<const float *>(
                &B[(tileOffset + innerRowOffsetB + innerRowB) * M +
                   (colOffsetB + innerColB * VECTOR_SIZE)]);
      } else if ((innerRowOffsetB + innerRowB) < BK &&
                 (innerColB * VECTOR_SIZE) < BM) {
        *reinterpret_cast<float *>(
            &Bs[innerRowOffsetB + innerRowB][innerColB * VECTOR_SIZE]) = 0.0f;
      }
    }
    if (threadIdx.x == numThreads - 1) { // load using different warp
      Ws[BN] = B_scale[(tileOffset / BLOCK_DIM) * M_scale +
                       (colOffsetB / BLOCK_DIM)];
    }

    __syncthreads();

    float b_scale = Ws[BN];
    floatx16 c = {0};
    for (size_t BKOffset = 0; BKOffset < BK; BKOffset += 16) {
      for (size_t wn = 0; wn < WITERN; ++wn) {
        for (size_t i = 0; i < 8; ++i) {
          a[wn][i] =
              As[BKOffset + warpY * 8 + i][warpRowOffset + wn * 32 + warpX];
        }
      }
      for (size_t wm = 0; wm < WITERM; ++wm) {
        for (size_t i = 0; i < 8; ++i) {
          b[wm][i] =
              Bs[BKOffset + warpY * 8 + i][warpColOffset + wm * 32 + warpX];
        }
      }
      for (size_t wn = 0; wn < WITERN; ++wn) {
        for (size_t wm = 0; wm < WITERM; ++wm) {
          c = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(
              *reinterpret_cast<long *>(a[wn]),
              *reinterpret_cast<long *>(b[wm]), c, 0, 0, 0);
        }
      }
    }
    for (size_t wn = 0; wn < WITERN; ++wn) {
      for (size_t wm = 0; wm < WITERM; ++wm) {
        for (size_t j = 0; j < 4; ++j) {
          for (size_t i = 0; i < 4; ++i) {
            d[wn][wm][i + j * 4] +=
                c[i + j * 4] *
                Ws[warpRowOffset + wn * 32 + j * 8 + warpY * 4 + i] * b_scale;
          }
        }
      }
    }

    __syncthreads();
  }

  for (size_t wn = 0; wn < WITERN; ++wn) {
    for (size_t wm = 0; wm < WITERM; ++wm) {
      for (size_t j = 0; j < 4; ++j) {
        for (size_t i = 0; i < 4; ++i) {
          if ((rowOffsetC + warpRowOffset + wn * 32 + j * 8 + warpY * 4 + i) <
                  N &&
              (colOffsetC + warpColOffset + wm * 32 + warpX) < M)
            C[(rowOffsetC + warpRowOffset + wn * 32 + j * 8 + warpY * 4 + i) *
                  M +
              (colOffsetC + warpColOffset + wm * 32 + warpX)] =
                (__hip_bfloat16)d[wn][wm][i + j * 4];
        }
      }
    }
  }
}

// Function to perform FP8 matrix multiplication
void fp8_mm(const __hip_fp8_e4m3_fnuz *A, const __hip_fp8_e4m3_fnuz *B,
            const float *A_scale, const float *B_scale, __hip_bfloat16 *C,
            size_t N, size_t K, size_t M) {
  
  const size_t BK = 128;
  const size_t BN = 128;
  const size_t BM = 128;
  const size_t WITERN = 1;
  const size_t WITERM = 1;
  
  dim3 numThreads((BN * BM) / (16 * WITERN * WITERM));
  dim3 numBlocks(cdiv(M, BM), cdiv(N, BN));
  
  fp8_mm_kernel<BN, BK, BM, WITERN, WITERM><<<numBlocks, numThreads>>>(
      A, B, A_scale, B_scale, C, N, K, M);
}

// Function to allocate memory on device
template <typename T>
T* allocateDeviceMemory(size_t size) {
  T* devicePtr;
  hipMalloc(&devicePtr, size * sizeof(T));
  return devicePtr;
}

// Function to free device memory
template <typename T>
void freeDeviceMemory(T* ptr) {
  hipFree(ptr);
}

// Function to copy memory from host to device
template <typename T>
void copyHostToDevice(T* dst, const T* src, size_t size) {
  hipMemcpy(dst, src, size * sizeof(T), hipMemcpyHostToDevice);
}

// Function to copy memory from device to host
template <typename T>
void copyDeviceToHost(T* dst, const T* src, size_t size) {
  hipMemcpy(dst, src, size * sizeof(T), hipMemcpyDeviceToHost);
}

// Function to initialize matrices with random values using seed
void initializeRandomMatrices(__hip_fp8_e4m3_fnuz* A, __hip_fp8_e4m3_fnuz* B, 
                              float* A_scale, float* B_scale, 
                              size_t N, size_t K, size_t M, int seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  
  // Initialize A
  for (size_t i = 0; i < N * K; i++) {
    A[i] = __hip_fp8_e4m3_fnuz(dist(gen));
  }
  
  // Initialize B
  for (size_t i = 0; i < K * M; i++) {
    B[i] = __hip_fp8_e4m3_fnuz(dist(gen));
  }
  
  // Initialize scaling factors
  for (size_t i = 0; i < N * cdiv(K, BLOCK_DIM); i++) {
    A_scale[i] = 1.0f;
  }
  
  for (size_t i = 0; i < cdiv(K, BLOCK_DIM) * cdiv(M, BLOCK_DIM); i++) {
    B_scale[i] = 1.0f;
  }
}

// Function to run a single benchmark test
double runBenchmark(const BenchmarkConfig& config, int numRuns = 10) {
  size_t N = config.n;
  size_t K = config.k;
  size_t M = config.m;
  int seed = config.seed;
  
  // Allocate host memory
  __hip_fp8_e4m3_fnuz* h_A = new __hip_fp8_e4m3_fnuz[N * K];
  __hip_fp8_e4m3_fnuz* h_B = new __hip_fp8_e4m3_fnuz[K * M];
  float* h_A_scale = new float[N * cdiv(K, BLOCK_DIM)];
  float* h_B_scale = new float[cdiv(K, BLOCK_DIM) * cdiv(M, BLOCK_DIM)];
  __hip_bfloat16* h_C = new __hip_bfloat16[N * M];
  
  // Initialize matrices
  initializeRandomMatrices(h_A, h_B, h_A_scale, h_B_scale, N, K, M, seed);
  
  // Allocate device memory
  __hip_fp8_e4m3_fnuz* d_A = allocateDeviceMemory<__hip_fp8_e4m3_fnuz>(N * K);
  __hip_fp8_e4m3_fnuz* d_B = allocateDeviceMemory<__hip_fp8_e4m3_fnuz>(K * M);
  float* d_A_scale = allocateDeviceMemory<float>(N * cdiv(K, BLOCK_DIM));
  float* d_B_scale = allocateDeviceMemory<float>(cdiv(K, BLOCK_DIM) * cdiv(M, BLOCK_DIM));
  __hip_bfloat16* d_C = allocateDeviceMemory<__hip_bfloat16>(N * M);
  
  // Copy data from host to device
  copyHostToDevice(d_A, h_A, N * K);
  copyHostToDevice(d_B, h_B, K * M);
  copyHostToDevice(d_A_scale, h_A_scale, N * cdiv(K, BLOCK_DIM));
  copyHostToDevice(d_B_scale, h_B_scale, cdiv(K, BLOCK_DIM) * cdiv(M, BLOCK_DIM));
  
  // Create HIP events for timing
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  
  // Warmup run
  fp8_mm(d_A, d_B, d_A_scale, d_B_scale, d_C, N, K, M);
  hipDeviceSynchronize();
  
  // Run multiple times for reliable timing
  std::vector<float> timings;
  for (int i = 0; i < numRuns; i++) {
    // Record the start event
    hipEventRecord(start, nullptr);
    
    // Perform matrix multiplication
    fp8_mm(d_A, d_B, d_A_scale, d_B_scale, d_C, N, K, M);
    
    // Record the stop event
    hipEventRecord(stop, nullptr);
    hipEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);
    timings.push_back(milliseconds);
  }
  
  // Calculate mean execution time in microseconds
  double meanMicroseconds = 0.0;
  for (float time : timings) {
    meanMicroseconds += time * 1000.0;  // Convert ms to μs
  }
  meanMicroseconds /= numRuns;
  
  // Clean up
  delete[] h_A;
  delete[] h_B;
  delete[] h_A_scale;
  delete[] h_B_scale;
  delete[] h_C;
  
  freeDeviceMemory(d_A);
  freeDeviceMemory(d_B);
  freeDeviceMemory(d_A_scale);
  freeDeviceMemory(d_B_scale);
  freeDeviceMemory(d_C);
  
  hipEventDestroy(start);
  hipEventDestroy(stop);
  
  return meanMicroseconds;
}

int main() {
  // Print device information
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, 0);
  std::cout << "Device: " << deviceProp.name << std::endl;
  std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
  
  // Define the benchmark configurations
  std::vector<BenchmarkConfig> benchmarks = {
    {7168, 1024, 1536, 8135},
    {1536, 1024, 3072, 6251},
    {7168, 1024, 576, 12346},
    {256, 1024, 7168, 5364},
    {2048, 1024, 7168, 6132},
    {7168, 1024, 4608, 7531},
    {2304, 1024, 7168, 12345},
    {7168, 1024, 512, 6563},
    {512, 1024, 4096, 17512},
    {7168, 6144, 1536, 6543},
    {1536, 6144, 3072, 234},
    {7168, 6144, 576, 9863},
    {256, 6144, 7168, 764243},
    {2048, 6144, 7168, 76547},
    {7168, 6144, 4608, 65436},
    {2304, 6144, 7168, 452345},
    {7168, 6144, 512, 12341},
    {512, 6144, 4096, 45245}
  };
  
  // Run benchmarks
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "+--------+--------+--------+----------+------------------+\n";
  std::cout << "|      K |      M |      N |     Seed | Time (μs)        |\n";
  std::cout << "+--------+--------+--------+----------+------------------+\n";
  
  double totalTime = 0.0;
  int count = 0;
  
  for (const auto& benchmark : benchmarks) {
    double time = runBenchmark(benchmark);
    totalTime += time;
    count++;
    
    std::cout << "| " << std::setw(6) << benchmark.k 
              << " | " << std::setw(6) << benchmark.m 
              << " | " << std::setw(6) << benchmark.n 
              << " | " << std::setw(8) << benchmark.seed 
              << " | " << std::setw(16) << time << " |\n";
  }
  
  std::cout << "+--------+--------+--------+----------+------------------+\n";
  std::cout << "| Mean execution time across all benchmarks: " << std::setw(10) << totalTime / count << " μs |\n";
  std::cout << "+----------------------------------------------------+\n";
  
  // Calculate total FLOPs across all benchmarks
  double totalFlops = 0.0;
  for (const auto& benchmark : benchmarks) {
    // Each matrix multiply performs 2*N*K*M FLOPs (multiply-add counts as 2)
    totalFlops += 2.0 * benchmark.n * benchmark.k * benchmark.m;
  }
  
  // Calculate mean TFLOPS
  double meanGFLOPs = (totalFlops / (totalTime * count)) * 0.001; // μs to ms, then to GFLOPS
  std::cout << "| Mean performance: " << std::setw(29) << meanGFLOPs << " GFLOPS |\n";
  std::cout << "+----------------------------------------------------+\n";
  
  return 0;
}
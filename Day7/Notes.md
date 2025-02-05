
# **Optimized CUDA GEMM (General Matrix Multiply)**

## **Overview**
This project implements an optimized **GEMM (General Matrix Multiplication)** kernel using CUDA. The code utilizes **shared memory tiling**, **loop unrolling**, and **parallel execution** for high performance. A CPU version is included for **verification and benchmarking**.

## **Key Features**
- **Shared Memory Optimization:** Reduces global memory accesses.
- **Tiling Strategy:** Uses **64×64×8** block sizes to balance memory and compute efficiency.
- **Loop Unrolling & Synchronization:** Improves data reuse and performance.
- **CUDA Events for Timing:** Measures execution time and computes GFLOPS.
- **CPU Reference GEMM:** Ensures correctness of GPU results.

---

## **Code Breakdown**
### **1. CUDA Kernel (`optimized_gemm_kernel`)**
```cpp
__global__ void optimized_gemm_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
    const int M, const int N, const int K
) {
    __shared__ float smem_A[64][8], smem_B[8][64];
    int row = blockIdx.y * 64 + threadIdx.y;
    int col = blockIdx.x * 64 + threadIdx.x;
    float acc = 0.0f;

    for (int k = 0; k < K; k += 8) {
        if (row < M && k + threadIdx.x < K) smem_A[threadIdx.y][threadIdx.x] = A[row * K + k + threadIdx.x];
        if (col < N && k + threadIdx.y < K) smem_B[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 8; ++i) acc += smem_A[threadIdx.y][i] * smem_B[i][threadIdx.x];

        __syncthreads();
    }
    
    if (row < M && col < N) C[row * N + col] = acc;
}
```
✅ **Optimized for memory efficiency**  
✅ **Parallel execution across CUDA threads**  
✅ **Tiling & shared memory for reduced latency**  

---

### **2. CPU Reference Implementation (`cpu_gemm`)**
```cpp
void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < K; k++)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}
```
✅ **Baseline for correctness verification**  

---

### **3. CUDA Error Checking Utility**
```cpp
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
```
✅ **Ensures robust error handling for CUDA API calls**

---

### **4. Execution & Performance Measurement**
```cpp
cudaEventRecord(start);
optimized_gemm_kernel<BM, BN, BK><<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
cudaEventRecord(stop);
cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
```
✅ **Uses CUDA Events for precise performance measurement**  
✅ **Computes GFLOPS & compares GPU vs. CPU accuracy**  

---

## **Compilation & Execution**
### **Build:**
```bash
nvcc -O3 -o optimized_gemm optimized_gemm.cu
```
### **Run:**
```bash
./optimized_gemm
```

## **Performance & Verification**
- ✅ **GFLOPS Calculation:** Measures computational efficiency.
- ✅ **Error Check:** Compares GPU results with CPU reference.
- ✅ **Time Measurement:** Reports execution time in milliseconds.

---

## **License**
MIT License. See [LICENSE](LICENSE) for details.

---

This README provides a precise and structured explanation of your CUDA GEMM implementation. 🚀

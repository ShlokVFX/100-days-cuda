## Code Explanation

### Key Definitions

```cpp
#define N (1 << 26)        // 67 million elements
#define THREADS_PER_BLOCK 256
#define ITERATIONS 100     // Run multiple iterations for averaging
```

- **N**: Defines the size of the array to be processed (67 million elements in this case).
- **THREADS_PER_BLOCK**: Defines the number of threads per block in the CUDA kernel. 256 is a commonly used number for many CUDA operations.
- **ITERATIONS**: Specifies how many iterations to run each benchmark, helping to average the results for accuracy.

### Custom Kernel: `sumReductionOptimized`

```cpp
__global__ void sumReductionOptimized(float *input, float *output, int n) {
    __shared__ float shared[THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float sum = 0.0f;
    
    if (idx < n) sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
    
    shared[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = shared[0];
}
```

- This **CUDA kernel** performs a parallel reduction of the input array in blocks. Each thread adds elements in a range, and shared memory is used for efficient intra-block communication.
- The `__syncthreads()` function ensures that all threads in a block synchronize before proceeding to the next step.
- After the reduction is complete within a block, the result is written to the output array.

### Final Reduction on CPU

```cpp
float finalReductionOnCPU(float *d_output, int size) {
    float *h_partial_sums = new float[size];
    cudaMemcpy(h_partial_sums, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    float total_sum = 0.0f;
    for (int i = 0; i < size; i++) {
        total_sum += h_partial_sums[i];
    }
    delete[] h_partial_sums;
    return total_sum;
}
```

- This function reduces the partial results from the GPU (each block’s sum) on the CPU side. It copies the output from the GPU and sums them up to get the final result.

### Benchmarking with CUDA Events

```cpp
cudaEvent_t startEvent, stopEvent;
cudaEventCreate(&startEvent);
cudaEventCreate(&stopEvent);
```

- **CUDA events** are used for precise timing of GPU operations. This allows us to measure the elapsed time more accurately than using CPU timers.

```cpp
float total_time_custom = 0.0f;
for (int i = 0; i < ITERATIONS; i++) {
    cudaEventRecord(startEvent, 0);
    sumReductionOptimized<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, N);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float ms;
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    total_time_custom += ms;
}
```

- We perform multiple iterations of the custom kernel (`ITERATIONS`), recording the start and stop times using CUDA events. This helps to average out the noise and provide more accurate results.

### Benchmarking with cuBLAS

```cpp
cublasHandle_t handle;
cublasCreate(&handle);
float dummy;
cublasSasum(handle, N, d_input, 1, &dummy);
```

- Before timing, we initialize the cuBLAS library and run a warm-up call to avoid any initialization overhead during benchmarking.

```cpp
float total_time_cublas = 0.0f;
for (int i = 0; i < ITERATIONS; i++) {
    cudaEventRecord(startEvent, 0);
    cublasSasum(handle, N, d_input, 1, &cublas_result);
    cudaEventRecord(stopEvent, 0);
    cudaDeviceSynchronize();
    cudaEventSynchronize(stopEvent);
    float ms;
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    total_time_cublas += ms;
}
```

- Similar to the custom kernel, we benchmark cuBLAS by running multiple iterations and recording the time.

### Calculating GFLOPS

```cpp
double gflops_custom = (N / seconds_custom) / 1e9;
double gflops_cublas = (N / seconds_cublas) / 1e9;
```

- **GFLOPS** (Giga Floating Point Operations per Second) is calculated as the total number of operations (N additions for sum reduction) divided by the time taken, normalized to billions of floating-point operations.

### Performance Comparison

```cpp
if (fabs(custom_result - cublas_result) / fabs(cublas_result) < 1e-3) {
    double speedup_percentage = (gflops_custom / gflops_cublas) * 100;
    std::cout << "Performance Comparison: " << std::endl;
    std::cout << "cuBLAS Speed: 100%" << std::endl;
    std::cout << "Custom Reduction Speed: " << speedup_percentage << "%" << std::endl;
}
```

- A **relative tolerance check** (`1e-3`) ensures the results from both methods are close enough before comparing performance. If the results are similar, the speedup of the custom kernel over cuBLAS is printed.


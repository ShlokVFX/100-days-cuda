# CUDA Learning Notes

## Day 1: Understanding CUDA Basics

### Introduction
This journey is about learning CUDA from scratch, building a solid foundation, and progressively writing efficient parallel programs. My first CUDA program is a simple **Vector Addition** kernel.

---

## Understanding CUDA Fundamentals
### 1. Why CUDA?
CUDA (Compute Unified Device Architecture) is a parallel computing framework by NVIDIA that allows us to use GPUs for general-purpose computation. It enables massive parallel execution, making it ideal for high-performance tasks.

### 2. Key CUDA Concepts
- **Host vs. Device:** The CPU (Host) and the GPU (Device) have separate memory spaces.
- **Threads & Blocks:** CUDA programs execute in a grid of threads organized into blocks.
- **Memory Management:** Data must be transferred between Host and Device memory.

---

## Writing the First CUDA Program: Vector Addition

### Step 1: Setting Up the Environment
Ensure CUDA is installed and configured properly. Verify with:
```sh
nvcc --version
```

### Step 2: Writing the Code
The goal is to perform element-wise addition of two vectors using the GPU.

#### **Kernel Function (Device Code)**
```cpp
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
```
- `__global__` defines the function as a CUDA kernel, executable on the GPU.
- `blockIdx.x`, `blockDim.x`, and `threadIdx.x` define thread indexing.
- Each thread computes one element of the result array.

#### **Host Code (CPU Code to Manage GPU Execution)**
```cpp
int main() {
    int N = 1000;  // Number of elements
    size_t size = N * sizeof(float);

    // Allocate memory on Host
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }
```
- Allocate and initialize memory for vectors on the CPU.

#### **Allocating Memory on the Device (GPU)**
```cpp
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
```
- `cudaMalloc` is used to allocate memory on the GPU.

#### **Copying Data from Host to Device**
```cpp
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
```
- `cudaMemcpy` transfers data between host and device memory.

#### **Launching the Kernel**
```cpp
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
```
- `<<<blocks, threads>>>` configures parallel execution.
- The grid is divided into blocks, and each block contains multiple threads.

#### **Copying the Result Back to Host**
```cpp
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
```

#### **Freeing Memory**
```cpp
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
```

---

## Key Takeaways
1. **Thread Hierarchy**: CUDA organizes execution using **Grids → Blocks → Threads**.
2. **Memory Transfer**: Explicit data movement is required between CPU (Host) and GPU (Device).
3. **Kernel Execution**: Each thread executes a part of the computation in parallel.
4. **Optimization Considerations**: The choice of threads per block and total blocks affects performance.

---

## Next Steps
- Experimenting with different block and thread configurations.
- Learning shared memory and reducing memory transfer overhead.
- Exploring more complex CUDA operations beyond vector addition.

This is just the beginning of my **100 Days of Learning CUDA Kernels** journey. 🚀


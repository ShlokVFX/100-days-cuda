# Optimized Vector Addition with FP16 and Float2

## Overview
This CUDA program implements optimized vector addition using `half2` (FP16) and `float2` (FP32) data types. It leverages vectorized operations and memory optimizations for improved performance on NVIDIA GPUs.

## Features
- **Vectorized Operations**: Uses `half2` for FP16 operations and `float2` for FP32 operations to process two elements at a time.
- **Memory Optimizations**: Utilizes `__ldg()` for cached reads to improve memory access efficiency.
- **Error Handling**: Includes CUDA error checking to ensure safe execution.
- **Alignment Checks**: Prevents misaligned memory access for `half2` operations.

## Code Explanation (Line by Line)

### 1. Includes and Definitions
```cpp
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>  // Fix for uintptr_t
#include <cstdio>   // Fix for printf
```
- Includes necessary CUDA headers for FP16 operations and runtime functionality.
- Includes `<cstdint>` for handling memory alignment using `uintptr_t`.
- Includes `<cstdio>` for debugging via `printf`.

### 2. Define a Device Function for Half2 Addition
```cpp
__device__ inline half2 my_hadd2(half2 a, half2 b) {
    return __hadd2(a, b);
}
```
- This function performs vectorized FP16 addition using the built-in `__hadd2()` function.
- `__device__` means it runs on the GPU.

### 3. FP16 Optimized Vector Addition Kernel
```cpp
__global__ void vector_add_half2(const half2* __restrict__ d_input1,
                                 const half2* __restrict__ d_input2,
                                 half2* __restrict__ d_output,
                                 size_t n_half2)
```
- `__global__` marks this function as a CUDA kernel.
- Uses `__restrict__` to optimize memory access.
- Inputs and outputs are in `half2` format to perform FP16 operations efficiently.

Inside the kernel:
```cpp
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_half2) {
        half2 a = __ldg(&d_input1[idx]);
        half2 b = __ldg(&d_input2[idx]);
```
- Computes global thread index `idx`.
- Uses `__ldg()` for cached reads, improving memory efficiency.

```cpp
        half2 c = my_hadd2(a, b);
        d_output[idx] = c;
```
- Calls `my_hadd2()` for FP16 addition and stores the result.

### 4. FP32 Optimized Vector Addition Kernel
```cpp
__global__ void vector_add_vec2(const float2* __restrict__ d_input1,
                                const float2* __restrict__ d_input2,
                                float2* __restrict__ d_output,
                                size_t n_vec2)
```
- Similar to `vector_add_half2`, but processes `float2` values.

Inside the kernel:
```cpp
    float2 a = __ldg(&d_input1[idx]);
    float2 b = __ldg(&d_input2[idx]);
```
- Uses `__ldg()` for efficient memory access.

```cpp
    float2 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    d_output[idx] = c;
```
- Computes vectorized addition manually for `float2`.

```cpp
    if (idx < n_vec2 - 1) {
        float2 d;
        d.x = a.x * b.x;
        d.y = a.y * b.y;
        d_output[idx + 1] = d;
    }
```
- Optionally stores the multiplication result to demonstrate vectorized operations.

### 5. Solution Function
```cpp
extern "C" void solution(float* d_input1, float* d_input2, float* d_output, size_t n) {
```
- Extern "C" allows interoperability with C code.
- This function handles kernel launches dynamically based on input size.

```cpp
    size_t n_vec2 = n / 2;
    size_t remainder = n % 2;
```
- Determines how many `float2` elements can be processed.
- Computes the remainder for `half2` processing.

```cpp
    int threadsPerBlock = (n < 10'000'00) ? 256 : 512;
    int blocksPerGrid = (n_vec2 + threadsPerBlock - 1) / threadsPerBlock;
```
- Chooses an optimal block size based on input size.

#### Launching the FP32 Kernel
```cpp
    if (n_vec2 > 0) {
        vector_add_vec2<<<blocksPerGrid, threadsPerBlock>>>(
            reinterpret_cast<const float2*>(d_input1),  
            reinterpret_cast<const float2*>(d_input2),
            reinterpret_cast<float2*>(d_output),
            n_vec2);
    }
```
- If `n_vec2 > 0`, launches `vector_add_vec2`.
- Uses `reinterpret_cast` to safely cast memory types.

#### Handling Remainder with FP16 Kernel
```cpp
    if (remainder > 0) {
        size_t offset = n - remainder;
        blocksPerGrid = (remainder + threadsPerBlock - 1) / threadsPerBlock;
```
- Computes offset for remaining elements.
- Adjusts grid size for remainder processing.

```cpp
        if ((reinterpret_cast<std::uintptr_t>(d_input1 + offset) % 4) == 0) {
            vector_add_half2<<<blocksPerGrid, threadsPerBlock>>>(
                reinterpret_cast<const half2*>(d_input1 + offset),
                reinterpret_cast<const half2*>(d_input2 + offset),
                reinterpret_cast<half2*>(d_output + offset),
                remainder / 2);
        } else {
            printf("🚨 Error: Memory misaligned for half2 access!\n");
        }
```
- Ensures `half2` alignment (4 bytes) before launching `vector_add_half2`.
- Prints an error message if memory is misaligned.

### 6. CUDA Error Handling
```cpp
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Error: %s\n", cudaGetErrorString(err));
    }
```
- Checks for kernel launch errors using `cudaGetLastError()`.

## Usage
### Compilation
```sh
nvcc -arch=sm_75 -o vector_add vector_add.cu
```
- Compiles the CUDA program.

### Execution
```sh
./vector_add
```
- Runs the program.

## Performance Considerations
- Uses vectorized operations for efficiency.
- Ensures proper memory access patterns.

## License
MIT License.

## Author
Part of **100 Days of Learning CUDA Kernels** challenge.



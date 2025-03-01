# CUDA Vector Addition Optimization - README

## Overview
This CUDA program performs element-wise vector addition using two different kernel implementations:
1. **`vector_add`**: Processes individual `float` elements.
2. **`vector_add_vec2`**: Processes `float2` elements for potential memory coalescing benefits.

The program dynamically selects the appropriate kernel based on the input size and ensures proper handling of odd-sized vectors.

---

## Code Explanation

### 1. **Header Inclusion**
```cpp
#include <cuda_runtime.h>
```
This includes CUDA runtime API functions and data types, which are necessary for device memory management and kernel execution.

---

### 2. **Kernel: `vector_add` (Standard Float Addition)**
```cpp
__global__ void vector_add(const float* __restrict__ d_input1,
                           const float* __restrict__ d_input2,
                           float* __restrict__ d_output,
                           size_t n)
```
- **`__global__`**: Marks this function as a CUDA kernel to be executed on the GPU.
- **`__restrict__`**: Informs the compiler that pointers don’t alias, enabling better memory optimization.
- **Parameters:**
  - `d_input1`: Pointer to the first input vector.
  - `d_input2`: Pointer to the second input vector.
  - `d_output`: Pointer to the output vector.
  - `n`: Total number of elements in the vector.

#### **Thread Index Calculation**
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```
Each thread computes its unique index based on:
- `blockIdx.x`: Block index.
- `blockDim.x`: Number of threads per block.
- `threadIdx.x`: Thread index within the block.

#### **Bounds Check & Computation**
```cpp
if (idx < n) {
    d_output[idx] = d_input1[idx] + d_input2[idx];
}
```
- Ensures the thread index does not exceed array bounds.
- Performs element-wise addition for corresponding indices.

---

### 3. **Kernel: `vector_add_vec2` (Optimized Using `float2` for Memory Coalescing)**
```cpp
__global__ void vector_add_vec2(const float* __restrict__ d_input1,
                                const float* __restrict__ d_input2,
                                float* __restrict__ d_output,
                                size_t n_vec2)
```
- Similar to `vector_add`, but operates on `float2` elements instead of `float`.
- `n_vec2` represents the number of `float2` pairs (half the original size).

#### **Reading and Computing with `float2`**
```cpp
float2 a = reinterpret_cast<const float2*>(d_input1)[idx];
float2 b = reinterpret_cast<const float2*>(d_input2)[idx];
float2 c;
c.x = a.x + b.x;
c.y = a.y + b.y;
```
- Uses `reinterpret_cast` to treat input arrays as `float2`.
- Loads two `float` values at once, potentially improving memory efficiency.
- Adds corresponding `x` and `y` components of the `float2` values.

#### **Writing `float2` Results Back to Memory**
```cpp
reinterpret_cast<float2*>(d_output)[idx] = c;
```
- Stores the computed `float2` result in the output array.
- Reduces memory transactions by processing two floats at a time.

---

### 4. **Host Function: `solution` (Kernel Invocation & Load Balancing)**
```cpp
extern "C" void solution(float* d_input1, float* d_input2, float* d_output, size_t n)
```
- `extern "C"` ensures correct function name linking in C++ code.
- Orchestrates the kernel execution and optimally distributes computation.

#### **Handling `float2` and Remainder Processing**
```cpp
size_t n_vec2 = n / 2;
size_t remainder = n % 2;
```
- `n_vec2` stores the number of `float2` elements that can be processed.
- `remainder` keeps track of any leftover elements that require standard `float` processing.

#### **Thread & Block Configuration**
```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (n_vec2 + threadsPerBlock - 1) / threadsPerBlock; // Fixed grid size calculation
```
- Defines `threadsPerBlock` as 256, a common optimal choice for GPU workloads.
- Computes the number of required blocks using integer ceiling division.

#### **Kernel Execution: `vector_add_vec2` (Optimized Processing)**
```cpp
if (n_vec2 > 0) {
    vector_add_vec2<<<blocksPerGrid, threadsPerBlock>>>(d_input1, d_input2, d_output, n_vec2);
}
```
- Launches `vector_add_vec2` if there are sufficient `float2` elements.
- Uses previously calculated `blocksPerGrid` and `threadsPerBlock`.

#### **Handling the Remainder with `vector_add`**
```cpp
if (remainder > 0) {
    size_t offset = n - remainder;
    blocksPerGrid = (remainder + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_input1 + offset, d_input2 + offset, d_output + offset, remainder);
}
```
- If `remainder > 0`, calculates an `offset` to process the remaining `float` elements.
- Recomputes `blocksPerGrid` for the leftover elements.
- Launches `vector_add` for the remainder.

---

## **Performance Considerations**
- **Memory Coalescing:**
  - `vector_add_vec2` optimizes memory access by using `float2`.
  - Helps with efficient global memory transactions.
- **Occupancy:**
  - Uses 256 threads per block, a good balance for GPU occupancy.
- **Load Balancing:**
  - Dynamically determines whether to use `vector_add_vec2` or `vector_add`.
- **Branch Divergence:**
  - Minimal since all threads within a warp execute the same operation.

## **Conclusion**
This CUDA program efficiently adds two vectors using both standard `float` and optimized `float2` techniques. By leveraging memory coalescing and dynamic kernel selection, it achieves higher performance on modern GPUs.


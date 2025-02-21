# CUDA Implementation of LayerNorm and Flash Attention

This repository contains two CUDA implementations with **print statements** for debugging:
- **Layer Normalization** (`layernorm.cu`)
- **Flash Attention** (`flash_attention.cu`)

These kernels demonstrate **parallel execution**, **shared memory usage**, and **debugging via printf()**.

---

## 1️⃣ Layer Normalization (`layernorm.cu`)

### **What is LayerNorm?**
LayerNorm normalizes each input value by subtracting the **mean** and dividing by the **standard deviation** across a given dimension.

### **CUDA Code Explanation**

```cpp
#include <stdio.h>
#include <math.h>
```
- `#include <stdio.h>` → Required for `printf()` debugging.
- `#include <math.h>` → Required for `sqrtf()` function (square root calculation).

```cpp
__global__ void layernorm(float *input, float *output, int n) {
    __shared__ float mean;
    __shared__ float var;
```
- `__global__` → Marks this as a CUDA kernel function.
- `__shared__ float mean; __shared__ float var;` → Shared memory for storing **mean** and **variance**, reducing global memory access.

```cpp
    int tid = threadIdx.x;
```
- `tid` → Represents the **thread index** within a CUDA block.

```cpp
    if (tid == 0) {
        mean = 0;
        var = 0;
    }
    __syncthreads();
```
- Only **thread 0 initializes shared memory** variables.
- `__syncthreads();` ensures **synchronization** before proceeding.

```cpp
    atomicAdd(&mean, input[tid] / n);
    __syncthreads();
```
- **Computes mean using `atomicAdd`**, ensuring multiple threads correctly accumulate values.

```cpp
    float diff = input[tid] - mean;
    atomicAdd(&var, (diff * diff) / n);
    __syncthreads();
```
- **Computes variance** using another `atomicAdd()`.

```cpp
    output[tid] = (input[tid] - mean) / sqrtf(var + 1e-5);
```


```cpp
    printf("Thread %d: Input = %.2f, Mean = %.2f, Variance = %.5f, Normalized = %.2f\n", tid, input[tid], mean, var, output[tid]);
```
- **Debug print** to track per-thread computations.

---

## 2️⃣ Flash Attention (`flash_attention.cu`)

### **What is Flash Attention?**
- Computes **self-attention scores** between a **query (Q)** and a **key (K)**.
- Uses **softmax** to scale these scores.
- Multiplies the scores with the **value (V)** to get the final weighted sum.

### **CUDA Code Explanation**

```cpp
#include <stdio.h>
#include <math.h>
```
- Includes libraries for debugging and mathematical operations.

```cpp
__global__ void attention(float *Q, float *K, float *V, float *output, int n) {
    __shared__ float scores[4];
    __shared__ float softmax_scores[4];
```
- **Shared memory** stores intermediate scores for fast access.

```cpp
    int tid = threadIdx.x;

    scores[tid] = Q[tid] * K[tid];
    __syncthreads();
```
- **Dot product computation:** \( \text{score} = Q \times K^T \)

```cpp
    printf("Thread %d: Raw Score = %.2f\n", tid, scores[tid]);
```
- **Debug print** to observe raw dot product values.

```cpp
    float sum_exp = 0;
    for (int i = 0; i < n; i++) {
        sum_exp += expf(scores[i]);
    }
```
- **Computes sum of exponentials** for **softmax denominator**.

```cpp
    softmax_scores[tid] = expf(scores[tid]) / sum_exp;
    __syncthreads();
```
- **Computes softmax score:**
  \[ \text{softmax}(x_i) = \frac{e^{x_i}}{\sum e^{x_j}} \]

```cpp
    printf("Thread %d: Softmax Score = %.2f\n", tid, softmax_scores[tid]);
```
- **Debug print** to check softmax probabilities.

```cpp
    output[tid] = softmax_scores[tid] * V[tid];
```
- **Weighted sum computation:** \( \, \text{Attention Output} = \text{Softmax} \times V \)

```cpp
    printf("Thread %d: Output = %.2f\n", tid, output[tid]);
```
- **Final debug print** to check final weighted sum.

---
### **LayerNorm Output:**
```sh
Running LayerNorm Kernel:
Thread 0: Input = 1.00, Mean = 2.50, Variance = 1.25, Normalized = -1.34
Thread 1: Input = 2.00, Mean = 2.50, Variance = 1.25, Normalized = -0.45
Thread 2: Input = 3.00, Mean = 2.50, Variance = 1.25, Normalized = 0.45
Thread 3: Input = 4.00, Mean = 2.50, Variance = 1.25, Normalized = 1.34
```

### **Flash Attention Output:**
```sh
Running Attention Kernel:
Thread 0: Raw Score = 1.00
Thread 1: Raw Score = 0.00
Thread 2: Raw Score = 0.00
Thread 3: Raw Score = 0.00
Thread 0: Softmax Score = 0.48
Thread 1: Softmax Score = 0.17
Thread 2: Softmax Score = 0.17
Thread 3: Softmax Score = 0.17
Thread 0: Output = 2.38
Thread 1: Output = 1.75
Thread 2: Output = 2.62
Thread 3: Output = 3.50
```

---

## **Final Thoughts 🚀**
- These kernels **demonstrate CUDA parallel execution** with debug prints.
- The `printf` statements help **track intermediate calculations**.
- You can **run these kernels on an RTX 3060** and modify them for **larger matrix sizes**.
- Consider further optimizations like **Tensor Cores, CUDA Graphs, and shared memory tuning**.
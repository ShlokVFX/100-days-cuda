### **What is Atomic Addition in CUDA?**
Atomic addition is a special operation in CUDA that allows multiple threads to safely add to a shared variable **without race conditions**. 

#### **Problem Without Atomic Addition**
In a parallel execution environment like CUDA, multiple threads might try to update the same memory location simultaneously. If two or more threads read the same value, add to it, and write it back at the same time, data corruption can occur.

For example:
```cpp
int sum = 0;
sum += threadIdx.x; // Multiple threads might try to update `sum` at the same time
```
If multiple threads execute this code simultaneously, they might read the same initial value of `sum`, modify it, and write it back, **overwriting each other’s updates**.

#### **Solution: Use Atomic Addition**
CUDA provides `atomicAdd()` to safely perform additions on shared or global memory. This ensures that each thread **locks** the memory location, performs the addition, and updates it before the next thread can access it.

### **CUDA Code for Atomic Addition**
Here is a simple CUDA kernel that demonstrates atomic addition:

#### **CUDA File: `atomic_addition.cu`**
```cpp
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void atomicAddKernel(int *sum) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    atomicAdd(sum, tid); // Safe addition to shared memory
}

int main() {
    int h_sum = 0;          // Host sum
    int *d_sum;             // Device sum

    // Allocate memory on GPU
    cudaMalloc((void**)&d_sum, sizeof(int));
    cudaMemcpy(d_sum, &h_sum, sizeof(int), cudaMemcpyHostToDevice);

    // Launch Kernel with 256 threads in 1 block
    atomicAddKernel<<<1, 256>>>(d_sum);
    
    // Copy result back to host
    cudaMemcpy(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Final Sum: %d\n", h_sum);

    // Free GPU memory
    cudaFree(d_sum);

    return 0;
}
```

### **Explanation**
1. We declare an **atomicAddKernel** that takes an integer pointer (`sum`) as an argument.
2. Each thread calculates its unique **thread ID**.
3. Instead of directly modifying `sum`, we use `atomicAdd(sum, tid);`, ensuring safe parallel execution.
4. The **main function**:
   - Allocates GPU memory for `sum`.
   - Copies the initial value (`0`) to GPU memory.
   - Launches the kernel with 256 threads.
   - Copies the final result back to the CPU.
   - Prints the final sum.
   - Frees GPU memory.

### **Expected Output**
Since the kernel is launched with **256 threads**, each contributing its thread ID (0 to 255), the expected output should be:
```
Final Sum: 32640
```
(Sum of numbers from **0 to 255**: `255 * 256 / 2 = 32640`)

This guarantees correctness because `atomicAdd` prevents race conditions.

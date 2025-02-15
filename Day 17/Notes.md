### **🚀 Summary of cuBLAS SAXPY Implementation**
This CUDA program uses **cuBLAS** to perform the SAXPY operation:  

where **α (factor)** is a scalar and **a, b** are vectors. The program also measures **GFLOPS** and computes **error metrics**.

---

### **📌 Code Breakdown**
#### **1️⃣ Header Inclusions**
```cpp
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>  // For fabs()
```
- Includes standard **I/O, memory, and CUDA libraries**.
- Uses **cuBLAS** for GPU-accelerated computation.
- Uses `<cmath>` for **floating-point comparisons**.

---

#### **2️⃣ Define Error Checking Macro**
```cpp
#define CHECK_CUBLAS(call) { ... }
```
- Wraps **cuBLAS API calls** to check for errors.

---

#### **3️⃣ Verification Function (Error Metrics)**
```cpp
void verify_result(float* a, float* b, float* c, float factor, int n) { ... }
```
- Computes **expected output**: `expected = factor * a[i] + b[i]`
- Calculates:
  - **Mean Absolute Error (MAE)**
  - **Max Relative Error**

---

#### **4️⃣ Main Function (CUDA Execution)**
```cpp
int main() { ... }
```
- Defines **vector size**:  
  ```cpp
  int n = 1 << 20;  // 1,048,576 elements
  ```
- Allocates **host memory** (`malloc()`).
- Initializes **random float values** in vectors.

---

#### **5️⃣ Allocate GPU Memory & Copy Data**
```cpp
cudaMalloc((void**)&d_a, bytes);
cudaMalloc((void**)&d_b, bytes);
cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);
```
- Allocates **device memory**.
- Copies **input vectors** to GPU.

---

#### **6️⃣ cuBLAS Setup & Performance Timing**
```cpp
cublasHandle_t handle;
CHECK_CUBLAS(cublasCreate(&handle));

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
```
- Creates a **cuBLAS handle**.
- Uses **CUDA events** to measure execution time.

---

#### **7️⃣ Perform SAXPY Using cuBLAS**
```cpp
CHECK_CUBLAS(cublasSaxpy(handle, n, &factor, d_a, 1, d_b, 1));
```
  using **cuBLAS SAXPY**.

---

#### **8️⃣ Compute Execution Time & GFLOPS**
```cpp
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
double gflops = (2.0 * n) / (milliseconds / 1000.0 * 1e9);
```
- Measures **execution time**.
- Computes **GFLOPS** using:
  \[
  \text{GFLOPS} = \frac{2n}{\text{time in seconds} \times 10^9}
  \]

---

#### **9️⃣ Copy Results Back to Host & Verify**
```cpp
cudaMemcpy(c, d_b, bytes, cudaMemcpyDeviceToHost);
verify_result(a, b, c, factor, n);
```
- Copies results **from GPU to CPU**.
- Runs **error verification**.

---

#### **🔟 Cleanup Resources**
```cpp
cublasDestroy(handle);
cudaFree(d_a);
cudaFree(d_b);
free(a);
free(b);
free(c);
```
- Frees **GPU & CPU memory**.
- Destroys **cuBLAS handle**.
- Destroys **CUDA event timers**.

---

### **🎯 Final Output**
✅ **Execution Time** (ms)  
✅ **GFLOPS**  
✅ **Error Metrics**  
✅ **Verification Passed!**

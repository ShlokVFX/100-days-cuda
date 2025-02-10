

### **3️⃣ Warp-Level Reduction Functions**
These functions perform **parallel reductions** inside a warp.

#### **Warp Reduction for Maximum Value**
```cpp
__device__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}
```
- Uses **`__shfl_down_sync()`** to efficiently compute the **maximum value** across a warp.

#### **Warp Reduction for Sum**
```cpp
__device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```
- Uses **`__shfl_down_sync()`** to efficiently **sum all values** within a warp.

---

### **4️⃣ Softmax Kernel (Row-wise)**
```cpp
__global__ void softmaxOptimized(float* input, float* output, int rows, int cols) {
```
- `__global__` → Defines a **CUDA kernel function** that runs on the GPU.
- **Parameters:**
  - `input` → Pointer to the input matrix.
  - `output` → Pointer to the output matrix (softmax results).
  - `rows, cols` → Dimensions of the matrix.

#### **Allocate Shared Memory**
```cpp
extern __shared__ float shared_data[];
```
- `extern __shared__` → Declares **dynamically allocated shared memory** for storing intermediate softmax computations.
- 
#### **Step 1: Compute Maximum for Numerical Stability**
```cpp
float max_val = -INFINITY;
for (int i = tid; i < cols; i += blockDim.x) {
    max_val = fmaxf(max_val, input[row * cols + i]);
}
```
- Each thread **reads a portion of the row** and finds the **maximum value**.

#### **Step 2: Block-wide Reduction for Maximum**
```cpp
__shared__ float block_max;
if (tid == 0) block_max = -INFINITY;
__syncthreads();
    
atomicMax((int*)&block_max, __float_as_int(max_val));
__syncthreads();
    
max_val = block_max;
```
- **Uses `atomicMax()`** to find the **global max** across the row.

#### **Step 3: Compute Exponentials and Sum**
```cpp
float sum = 0.0f;
for (int i = tid; i < cols; i += blockDim.x) {
    shared_data[i] = expf(input[row * cols + i] - max_val);
    sum += shared_data[i];
}
```
- Each thread **computes the exponentials** using `expf()` and **stores them in shared memory**.

#### **Step 4: Block-wide Reduction for Sum**
```cpp
__shared__ float block_sum;
if (tid == 0) block_sum = 0.0f;
__syncthreads();
    
atomicAdd(&block_sum, sum);
__syncthreads();
    
sum = block_sum;
```
- **Uses `atomicAdd()`** to compute the total sum of exponentials.

#### **Step 5: Normalize and Store Results**
```cpp
for (int i = tid; i < cols; i += blockDim.x) {
    output[row * cols + i] = shared_data[i] / sum;
}
```
- Each thread **divides exponentials by the sum** to compute softmax probabilities.

---

### **5️⃣ Host Function for Kernel Execution**
```cpp
void softmax(float* h_input, float* h_output, int rows, int cols) {
```
- **Handles memory allocation, data transfer, and kernel execution.**

#### **Allocate Device Memory**
```cpp
float *d_input, *d_output;
size_t size = rows * cols * sizeof(float);

cudaMalloc((void**)&d_input, size);
cudaMalloc((void**)&d_output, size);
```
- **`cudaMalloc()`** → Allocates GPU memory for input and output arrays.

#### **Copy Data to GPU**
```cpp
cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
```
- **Transfers the input matrix** from CPU → GPU.

#### **Launch CUDA Kernel**
```cpp
int shared_mem_size = cols * sizeof(float);
softmaxOptimized<<<rows, BLOCK_SIZE, shared_mem_size>>>(d_input, d_output, rows, cols);
```
- **Launches the kernel** with:
  - `rows` blocks (one per row).
  - `BLOCK_SIZE` threads per block.
  - `shared_mem_size` bytes of shared memory.

#### **Copy Results Back to CPU**
```cpp
cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
```
- **Transfers the softmax results** from GPU → CPU.

#### **Free Device Memory**
```cpp
cudaFree(d_input);
cudaFree(d_output);
```
- **Frees GPU memory** after execution.

---

### **6️⃣ Main Function (Test Case)**
```cpp
int main() {
```
- **Initializes test data** and calls `softmax()`.

#### **Define Input Matrix**
```cpp
const int rows = 2, cols = 4;
float h_input[rows * cols] = {1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0};
float h_output[rows * cols];
```

#### **Compute Softmax**
```cpp
softmax(h_input, h_output, rows, cols);

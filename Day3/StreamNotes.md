## **CUDA Streams in `gpu_vs_cpu_complex_math_benchmark.cu`**
CUDA **streams** play a crucial role in this program by enabling **concurrent execution** of memory transfers and kernel computations, leading to improved GPU performance.

---

## **1️⃣ What are CUDA Streams?**
A **CUDA Stream** is a sequence of operations (memory transfers, kernel launches, etc.) that execute **in order within the stream**, but operations in **different streams can run concurrently**.

By default, **all operations go into the same implicit stream** (called **the default stream**), which is **blocking**—meaning each operation must finish before the next one starts.

However, by explicitly **creating multiple streams**, we can **overlap memory transfers and computation**, leading to better utilization of the GPU.

---

## **2️⃣ Streams in This Program**
In **`gpu_vs_cpu_complex_math_benchmark.cu`**, we use **two CUDA streams** to **improve GPU performance**.

### **Declaring and Creating CUDA Streams**
```cpp
cudaStream_t stream1, stream2;
CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));
```
- **`stream1` and `stream2` are created** to manage **two independent execution paths**.
- They allow us to process two halves of the data **simultaneously**.

---

## **3️⃣ Where Do We Use Streams?**
We use streams to **split the workload** into two independent parts:

| **Step**  | **Operation** | **Stream Used** |
|-----------|-------------|----------------|
| **1** | Copy **first half** of `h_A` and `h_B` to GPU | `stream1` |
| **2** | Copy **second half** of `h_A` and `h_B` to GPU | `stream2` |
| **3** | Launch **kernel on first half** | `stream1` |
| **4** | Launch **kernel on second half** | `stream2` |
| **5** | Copy **first half** of `d_C` back to `h_C` | `stream1` |
| **6** | Copy **second half** of `d_C` back to `h_C` | `stream2` |
| **7** | Synchronize streams to ensure all tasks finish | `stream1 & stream2` |

### **(A) Using Streams for Memory Transfers**
Instead of copying all data sequentially, we use **asynchronous memory copies** (`cudaMemcpyAsync`) so that both halves of the data transfer happen **simultaneously** in different streams.

```cpp
CHECK_CUDA_ERROR(cudaMemcpyAsync(d_A, h_A, halfElements * sizeof(float), cudaMemcpyHostToDevice, stream1));
CHECK_CUDA_ERROR(cudaMemcpyAsync(d_B, h_B, halfElements * sizeof(float), cudaMemcpyHostToDevice, stream1));

CHECK_CUDA_ERROR(cudaMemcpyAsync(d_A + halfElements, h_A + halfElements, halfElements * sizeof(float), cudaMemcpyHostToDevice, stream2));
CHECK_CUDA_ERROR(cudaMemcpyAsync(d_B + halfElements, h_B + halfElements, halfElements * sizeof(float), cudaMemcpyHostToDevice, stream2));
```
### **Why?**
- **Without streams**, the CPU would wait for each memory copy to finish before launching the next one.
- **With streams**, the memory copies for `stream1` and `stream2` happen **concurrently**, saving time.

---

### **(B) Launching Kernels in Streams**
After transferring data, we launch the **complex math kernel** on each half of the dataset, using separate streams:

```cpp
vectorComplexCompute<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A, d_B, d_C, halfElements);
vectorComplexCompute<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_A + halfElements, d_B + halfElements, d_C + halfElements, halfElements);
```

### **Why?**
- The two kernel executions can **run in parallel** if the GPU supports concurrent execution.
- The GPU scheduler will **automatically assign different blocks to available SMs (Streaming Multiprocessors)**.
- **This doubles the computational throughput** by utilizing multiple SMs efficiently.

---

### **(C) Copying Results Back in Parallel**
Once the kernels finish executing, the results need to be copied **back to the host** (`h_C`). Again, we use **asynchronous memory copies**:

```cpp
CHECK_CUDA_ERROR(cudaMemcpyAsync(h_C, d_C, halfElements * sizeof(float), cudaMemcpyDeviceToHost, stream1));
CHECK_CUDA_ERROR(cudaMemcpyAsync(h_C + halfElements, d_C + halfElements, halfElements * sizeof(float), cudaMemcpyDeviceToHost, stream2));
```

### **Why?**
- **Without streams**: We would have to **wait** for the first half to be copied before the second half starts.
- **With streams**: The two memory copies **happen simultaneously**, reducing overall execution time.

---

### **(D) Synchronizing Streams**
Before verifying results, we **ensure all tasks have completed**:

```cpp
CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));
```

### **Why?**
- **`cudaStreamSynchronize(stream1)`** ensures that all operations (memory copies + kernel) in `stream1` are finished.
- **`cudaStreamSynchronize(stream2)`** does the same for `stream2`.

🚀 **Without synchronization, the CPU might try to read data before GPU computations finish, leading to errors.**

---

## **4️⃣ How Streams Improve Performance**
| **Without Streams (Sequential Execution)** | **With Streams (Concurrent Execution)** |
|---------------------------------|---------------------------------|
| Host copies **all data** to GPU first | Host copies **half data in `stream1`**, and **half in `stream2`** simultaneously |
| **Waits** for memory transfer to finish | **Overlaps** memory transfers with kernel execution |
| Kernel processes **entire dataset** sequentially | Kernel processes **half in `stream1`**, and **half in `stream2`** simultaneously |
| **Waits** for kernel execution to finish | **Overlaps kernel execution across streams** |
| **Waits** for result copy to finish | **Copies result back asynchronously** |

✅ **Using streams reduces idle time on the GPU**, leading to **faster execution**.  
✅ **Overlapping memory transfers and kernel execution improves throughput**.

---

## **5️⃣ Key Takeaways**
✅ **CUDA Streams allow concurrent execution** of memory transfers and computation.  
✅ **Memory transfers (`cudaMemcpyAsync`) and kernel launches (`<<<...>>>`) use separate streams (`stream1`, `stream2`).**  
✅ **Synchronization (`cudaStreamSynchronize`) ensures correct execution order.**  
✅ **Using multiple streams helps overlap computation and data movement, reducing execution time.**  

🚀 **Result:** **Higher performance compared to executing everything sequentially!**  

---

## **6️⃣ Visual Representation**
### **Without Streams (Sequential Execution)**
```
| Copy h_A to d_A | Copy h_B to d_B | Kernel Execution | Copy d_C to h_C |
--------------------------------------------------------------------> (Time)
```

### **With Streams (Concurrent Execution)**
```
| Copy h_A (1st Half) to d_A (stream1) | Kernel (stream1) | Copy d_C (1st Half) to h_C (stream1) |
| Copy h_B (2nd Half) to d_B (stream2) | Kernel (stream2) | Copy d_C (2nd Half) to h_C (stream2) |
--------------------------------------------------------------------------------------> (Time)
```
🚀 **Using two streams enables simultaneous execution, reducing overall time.**  

---

## **7️⃣ Final Summary**
🔹 **What did we do?**  
- Used **two streams** (`stream1`, `stream2`) to process two halves of the data **simultaneously**.  
- Overlapped **memory transfers and kernel execution** to minimize idle time.  
- Improved **GPU utilization**, leading to **faster execution**.

🔹 **Why does it matter?**  
- **Avoids blocking execution** on memory copies.  
- **Utilizes multiple SMs** to run kernels in parallel.  
- **Improves efficiency** by keeping the GPU busy.

🚀 **Now the GPU runs much faster than the CPU, thanks to streams!** 🚀  


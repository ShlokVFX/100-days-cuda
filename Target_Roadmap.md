# 🚀 100 Days of CUDA - Learning Plan

## 🏆 Goal
Master **CUDA, cuBLAS, cuDNN, Parallel Computing, Performance Optimization, and C++** by building optimized GPU programs over 100 days.

## 📅 Roadmap

### **Week 1-2: Foundations of CUDA & Parallel Computing**
🔹 **Goals:**
- Learn CUDA programming model, memory hierarchy, and basic kernel execution.

🔹 **Topics:**
✅ CUDA installation & setup  
✅ Basics of GPU architecture & memory types (Global, Shared, Local, Registers)  
✅ Writing first CUDA kernel (`__global__` functions)  
✅ Thread hierarchy: Blocks, Grids, Warps  
✅ Understanding execution model: SIMD, SIMT  

🔹 **Hands-on:**
- Implement **vector addition** using CUDA.
- Experiment with different **block and grid sizes**.
- Measure performance using **CUDA events**.

---
### **Week 3-4: Memory Management & Performance Optimization (Level 1)**
🔹 **Goals:**
- Learn CUDA memory hierarchy & optimization techniques.

🔹 **Topics:**
✅ Host vs Device Memory, Unified Memory  
✅ Global memory coalescing & bank conflicts  
✅ Shared memory & its impact on performance  
✅ Memory bandwidth considerations  

🔹 **Hands-on:**
- Implement **Matrix Multiplication using Global Memory**.
- Optimize it using **Shared Memory**.
- Compare performance (Global vs Shared).

---
### **Week 5-6: Advanced CUDA - Streams, Asynchronous Execution, Profiling**
🔹 **Goals:**
- Learn CUDA Streams & parallel execution techniques.

🔹 **Topics:**
✅ CUDA Streams & concurrent kernel execution  
✅ CUDA events for timing & profiling  
✅ Asynchronous memory transfers (`cudaMemcpyAsync`)  
✅ Profiling with **NVIDIA Nsight, nvprof**  

🔹 **Hands-on:**
- Implement **pipelined memory transfers** with `cudaMemcpyAsync`.
- Launch **multiple kernels in parallel** using Streams.

---
### **Week 7-8: cuBLAS - Accelerating Linear Algebra on GPU**
🔹 **Goals:**
- Learn how to use cuBLAS for BLAS (Basic Linear Algebra Subroutines).

🔹 **Topics:**
✅ Introduction to cuBLAS API  
✅ cuBLAS matrix-vector, matrix-matrix operations  
✅ Strided and batched matrix multiplications  
✅ Memory management in cuBLAS  

🔹 **Hands-on:**
- Implement **Matrix Multiplication using cuBLAS**.
- Compare cuBLAS vs naive CUDA implementation.
- Implement **Batch GEMM (General Matrix Multiplication)**.

---
### **Week 9-10: cuDNN - Accelerating Deep Learning on GPU**
🔹 **Goals:**
- Learn cuDNN for optimized deep learning operations.

🔹 **Topics:**
✅ Convolutions in cuDNN  
✅ cuDNN tensor descriptors & data layout  
✅ Pooling, Activation, Softmax  
✅ Performance tuning in cuDNN  

🔹 **Hands-on:**
- Implement **CNN Forward Pass using cuDNN**.
- Benchmark cuDNN convolutions against a naive implementation.

---
### **Week 11-12: Performance Optimization (Level 2)**
🔹 **Goals:**
- Advanced CUDA optimizations for performance tuning.

🔹 **Topics:**
✅ Register usage optimization  
✅ Occupancy calculator & warp-level programming  
✅ CUDA Cooperative Groups  
✅ Tensor Cores & mixed precision computing  

🔹 **Hands-on:**
- Optimize **matrix multiplication using Tensor Cores**.
- Implement **warp-shuffle based reduction**.

---
### **Week 13-14: Multi-GPU Programming**
🔹 **Goals:**
- Learn multi-GPU communication and workload distribution.

🔹 **Topics:**
✅ Multi-GPU architecture & peer-to-peer communication  
✅ `cudaMemcpyPeer()` and NVLink  
✅ Using NCCL for distributed computing  

🔹 **Hands-on:**
- Implement **data parallelism using multi-GPU**.
- Optimize **inter-GPU communication**.

---
### **Week 15: Putting It All Together - Final Optimization Challenges**
🔹 **Goals:**
- Apply all learned concepts to real-world applications.

🔹 **Projects & Challenges:**
✅ Optimize **large-scale matrix multiplication (multi-GPU + cuBLAS)**  
✅ Implement **efficient CNN training pipeline using cuDNN**  
✅ Profile & optimize an **end-to-end CUDA application**  

---
### **🎯 Final Milestone (Day 100)**
🚀 **Publish a summary report + code repo** showcasing:
- Performance benchmarks
- Optimized CUDA applications
- Lessons learned + best practices  

---
## **🔥 Bonus (If Time Permits)**
✅ Deep dive into **CUTLASS for custom GEMM optimizations**  
✅ Learn **CUDA Graphs for efficient kernel execution**  
✅ Experiment with **NVIDIA Triton for inference acceleration**  

---
## **💡 Summary**
- **Week 1-2** → CUDA Basics + Parallel Computing  
- **Week 3-4** → Memory Optimization  
- **Week 5-6** → Streams + Profiling  
- **Week 7-8** → cuBLAS  
- **Week 9-10** → cuDNN  
- **Week 11-12** → Advanced Optimization  
- **Week 13-14** → Multi-GPU Programming  
- **Week 15** → Final Projects & Optimization  

## 🏆 Goal
Master **CUDA, cuBLAS, cuDNN, Parallel Computing, Performance Optimization, and C++** by building optimized GPU programs over 100 days.

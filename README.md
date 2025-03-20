Mentor: https://github.com/hkproj/

Discord : https://discord.gg/4Tg4TkJQzE

Instructions: https://github.com/hkproj/100-days-of-cuda

# 100 Days of CUDA Learning

This repository documents my 100-day journey of learning CUDA programming, writing optimized kernels, and improving GPU performance.

| Day  | Link | Notes |
|------|--------------|-------|
| 1    |  [Vector Addition Kernel](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001)  | Learned basic CUDA syntax and kernel execution - Vector Addtion and printing Hello Cuda. |
| 2    | [Benchmarking Vector Add](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2002) | Explored about Benchmarking in Cuda with Vector Add. |
| 3    |  [Cuda Streams](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2003)|CUDA Stream is a sequence of operations (memory transfers, kernel launches, etc.) that execute in order within the stream, but operations in different streams can run concurrently. |
| 4    | [Unified Mem VectorAdd](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2004)  |  Unified Memory simplifies memory management by allowing the CPU and GPU to share the same memory space. |
| 5    |  [Tiled MatMul](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2005)  | Matrix Multiplication in CUDA using shared memory to optimize performance. Tiling improves memory access efficiency by reducing global memory accesses and leveraging shared memory for faster computation. |
| 6    |  [Matrix Transpose](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2006)  |Coalesced memory access refers to a pattern where multiple threads in a warp access consecutive memory locations, leading to efficient memory transactions. |
| 7    |  [Basic GEMM with Optimizations](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2007)  | Utilizes shared memory tiling, loop unrolling, and parallel execution for high performance.|
| 8    |  [WMMA (Tensor Core with Double buffering)](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2008)  | WMMA leverages specialized Tensor Cores on NVIDIA GPUs to accelerate matrix multiplications.|
| 9    |  [Speeds Comparisons Matmul](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2009)  | Naive vs Tiled vs Thread Tiling vs WMMA/Tensor Core | [
| 10   |  [Advance Profiling](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2010)  | Importance of CUDA Profiling, Using Nsight systems  |
| 11   |  [Cuda Basic Softmax](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2011)  | Understanding Softmax  Algorithm and implementing in Cuda  |
| 12   |  [Better Softmax](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2012)  | Optimizing Softmax  Algorithm and Benchmarking it |
| 13   |  [SoftMax FP16 Acceleration](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2013)  | Higher Speedup achieved when used FP16 tensor cores optimization |
| 14   |  [Tensor MatMul](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2014)  | Naive vs Tensor core Matmul |
| 15   |  [CUDA Graphs](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2015)  | Reduced Overhead , Improved Performance, Simplified Code |
| 16   |  [SoftMax SuperFast](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2016)  |Implemented Cuda Algorithm that uses CuDNN + CudaStreams with FP16 Accelaration|
| 17   |  [cuBLAS VectorAdd](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2017)  | cuBLAS to perform Vector Addition and Benchmarking it |
| 18  |  [cuBLAS MatrixMultiplication](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2015)  | cuBLAS matmul with cuRAND for random num generation and benchmarking it |
| 19  |  [Sum Reduction](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2019)  | Performs a parallel reduction of the input array in blocks. Each thread adds elements in a range, and shared memory is used for efficient intra-block communication. |
| 20 |  [1D/2D Convolution](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2020)  | 1D convolution is used primarily in signal processing. 2D convolution is used primarily in image processing|
| 21 |  [Triton](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2021)  | Working with Triton , used Tutorials from Triton Documentation to run VectorAdd , matmul and softmax kernel|
| 22 |  [Fused Softmax in Triton](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2022)  | Triton fused softmax implementation provides a highly efficient way to compute the softmax function on GPUs. By fusing multiple operations into a single kernel, it achieves better performance compared to traditional implementations.  |
| 23 |  [LayerNorm and Flash Attention](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2023)  | Basic layerNorm and FlashAttention implementation in Cuda  | 
| 24 |  [Profiling Errors Solving](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2024/LOW-OCCUPANCY)  | Solved Questions related to profiling. Created strategies, before and after examples with command line debugging tools, and optimization techniques for GPU performance tuning.|
| 25 |  [Blelloch Prefix Scan ](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2025)  | Blelloch Prefix Scan using shared memory for efficiency.Solved More question related to design and GPU architecture.|
| 26 |  [FFT with Profiling](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2026)  | Fast Fourier Transform (FFT) Using Shared Memory + Profiling |
| 27 |  [Matmul_naive](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2027)  |Hit learning block so just repeated writing Naive Matmul on LEETGPU.com .  | 





## Goals:
- Understand CUDA fundamentals.
- Write optimized and efficient GPU kernels.
- Explore memory hierarchy, warp scheduling, and Tensor Cores.
- Apply CUDA to deep learning and high-performance computing.


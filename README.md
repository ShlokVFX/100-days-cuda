Mentor: https://github.com/hkproj/

Discord : https://discord.gg/4Tg4TkJQzE

Instructions: https://github.com/hkproj/100-days-of-cuda

# 100 Days of CUDA Learning

This repository documents my 100-day journey of learning CUDA programming, writing optimized kernels, and improving GPU performance.

| Day  | Output Summary | Notes | Link |
|------|--------------|-------|------|
| 1    |  [Vector Addition Kernel](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Output.png)  | Learned basic CUDA syntax and kernel execution - Vector Addtion and printing Hello Cuda. | [Day 1](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Readme.md) |
| 2    | [Benchmarking Vector Add](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2002/BenchmarkVectorAdd.png) | Explored about Benchmarking in Cuda with Vector Add. | [Day 2](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2002/Readme.md) |
| 3    |  [Cuda Streams](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2003/CudaStreams_result.png) [AtomicAddtion](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2003/AtomicAdditionResult.png) |CUDA Stream is a sequence of operations (memory transfers, kernel launches, etc.) that execute in order within the stream, but operations in different streams can run concurrently. | [Day 3](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2003/Readme.md) |
| 4    | [Unified Mem VectorAdd](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2004/VectorAdd_withErrorCheck.png)  |  Unified Memory simplifies memory management by allowing the CPU and GPU to share the same memory space. | [Day 4](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2004/Readme.md) |
| 5    |  [Tiled MatMul](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2005/Output.png)  | Matrix Multiplication in CUDA using shared memory to optimize performance. Tiling improves memory access efficiency by reducing global memory accesses and leveraging shared memory for faster computation. | [Day 5](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2005/Readme.md) |
| 6    |  [Matrix Transpose](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2006/Output.png)  |Coalesced memory access refers to a pattern where multiple threads in a warp access consecutive memory locations, leading to efficient memory transactions. | [Day 6](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2006/Readme.md) |
| 7    |  [Basic GEMM with Optimizations](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2007/Basic%20GEMM.png)  | Utilizes shared memory tiling, loop unrolling, and parallel execution for high performance.| [Day 7](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2007/Readme.md) |
| 8    |  [WMMA (Tensor Core with Double buffering)](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2008/wmma_tensored.png)  | WMMA leverages specialized Tensor Cores on NVIDIA GPUs to accelerate matrix multiplications.| [Day 8](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2008/Readme.md) |
| 9    |  [Speeds Comparisons Matmul](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2009/Output.png)  | Naive vs Tiled vs Thread Tiling vs WMMA/Tensor Core | [Day 9](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2009/Readme.md) |
| 10   |  [Advance Profiling](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2010/MatMulProfiling.png)  | Importance of CUDA Profiling, Using Nsight systems  | [Day 10](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2010/Readme.md) |
| 11   |  [Cuda Basic Softmax](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2011/output.png)  | Understanding Softmax  Algorithm and implementing in Cuda  | [Day 11](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2011/Readme.md) |
| 12   |  [Better Softmax](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2012/output.png)  | Optimizing Softmax  Algorithm and Benchmarking it | [Day 12](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2012/Readme.md) |
| 13   |  [SoftMax FP16 Acceleration](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2013/Gflops.png)  | Higher Speedup achieved when used FP16 tensor cores optimization | [Day 13](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2013/Readme.md) |
| 14   |  [Tensor MatMul](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2014/output.png)  | Naive vs Tensor core Matmul | [Day 14](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2014/Readme.md) |
| 15   |  [CUDA Graphs](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2015/output.png)  | Reduced Overhead , Improved Performance, Simplified Code | [Day 15](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2015/Readme.md) |
| 16   |  [SoftMax SuperFast](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2016/output.png)  |Implemented Cuda Algorithm that uses CuDNN + CudaStreams with FP16 Accelaration| [Day 16](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2016/Readme.md) |
| 17   |  [cuBLAS VectorAdd](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2017/output.png)  | cuBLAS to perform Vector Addition and Benchmarking it | [Day 17](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2017/Readme.md) |
| 18  |  [cuBLAS MatrixMultiplication](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2015/output.png)  | cuBLAS matmul with cuRAND for random num generation and benchmarking it | [Day 18](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2018/Readme.md) |
| 19  |  [Sum Reduction](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2019/output.png)  | Performs a parallel reduction of the input array in blocks. Each thread adds elements in a range, and shared memory is used for efficient intra-block communication. | [Day 19](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2019/Readme.md) |
| 20 |  [1D/2D Convolution](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2020/output.png)  | 1D convolution is used primarily in signal processing. 2D convolution is used primarily in image processing| [Day 20](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2020/Readme.md) |
| 21 |  [Triton](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2021/output.png)  | Working with Triton , used Tutorials from Triton Documentation to run VectorAdd , matmul and softmax kernel| [Day 21](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2021/Readme.md) |
| 22 |  [Fused Softmax in Triton](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2022/output.png)  | Triton fused softmax implementation provides a highly efficient way to compute the softmax function on GPUs. By fusing multiple operations into a single kernel, it achieves better performance compared to traditional implementations.  | [Day 22](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2022/Readme.md) |
| 23 |  [LayerNorm and Flash Attention](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2023/output.png)  | Basic layerNorm and FlashAttention implementation in Cuda  | [Day 23](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2023/Readme.md) |
| 24 |  [Profiling Errors Solving](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2024/LOW-OCCUPANCY/Output.png)  | Solved Questions related to profiling. Created strategies, before and after examples with command line debugging tools, and optimization techniques for GPU performance tuning.| [Day 24](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2024/Readme.md) |
| 25 |  [Blelloch Prefix Scan ](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2025/Output.png)  | Blelloch Prefix Scan using shared memory for efficiency.Solved More question related to design and GPU architecture.| [Day 25](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2025/Readme.md) |
| 26 |  [FFT with Profiling](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2026/Output.png)  | Fast Fourier Transform (FFT) Using Shared Memory + Profiling | [Day 26](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2026/Readme.md) |







## Goals:
- Understand CUDA fundamentals.
- Write optimized and efficient GPU kernels.
- Explore memory hierarchy, warp scheduling, and Tensor Cores.
- Apply CUDA to deep learning and high-performance computing.


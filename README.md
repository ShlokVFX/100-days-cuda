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
| 11   |  [Vector Addition Kernel](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Output.png)  | Learned basic CUDA syntax and kernel execution - Vector Addtion and printing Hello Cuda. | [Day 1](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Readme.md) |
| 12   |  [Vector Addition Kernel](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Output.png)  | Learned basic CUDA syntax and kernel execution - Vector Addtion and printing Hello Cuda. | [Day 1](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Readme.md) |
| 13   |  [Vector Addition Kernel](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Output.png)  | Learned basic CUDA syntax and kernel execution - Vector Addtion and printing Hello Cuda. | [Day 1](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Readme.md) |
| 14   |  [Vector Addition Kernel](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Output.png)  | Learned basic CUDA syntax and kernel execution - Vector Addtion and printing Hello Cuda. | [Day 1](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Readme.md) |
| 15   |  [Vector Addition Kernel](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Output.png)  | Learned basic CUDA syntax and kernel execution - Vector Addtion and printing Hello Cuda. | [Day 1](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Readme.md) |
| 16   |  [Vector Addition Kernel](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Output.png)  | Learned basic CUDA syntax and kernel execution - Vector Addtion and printing Hello Cuda. | [Day 1](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Readme.md) |
| 17   |  [Vector Addition Kernel](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Output.png)  | Learned basic CUDA syntax and kernel execution - Vector Addtion and printing Hello Cuda. | [Day 1](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Readme.md) |
| 18   |  [Vector Addition Kernel](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Output.png)  | Learned basic CUDA syntax and kernel execution - Vector Addtion and printing Hello Cuda. | [Day 1](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Readme.md) |
| 19   |  [Vector Addition Kernel](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Output.png)  | Learned basic CUDA syntax and kernel execution - Vector Addtion and printing Hello Cuda. | [Day 1](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Readme.md) |
| 20   |  [Vector Addition Kernel](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Output.png)  | Learned basic CUDA syntax and kernel execution - Vector Addtion and printing Hello Cuda. | [Day 1](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Readme.md) |
| 21   |  [Vector Addition Kernel](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Output.png)  | Learned basic CUDA syntax and kernel execution - Vector Addtion and printing Hello Cuda. | [Day 1](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Readme.md) |
| 22   |  [Vector Addition Kernel](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Output.png)  | Learned basic CUDA syntax and kernel execution - Vector Addtion and printing Hello Cuda. | [Day 1](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Readme.md) |
| 23   |  [Vector Addition Kernel](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Output.png)  | Learned basic CUDA syntax and kernel execution - Vector Addtion and printing Hello Cuda. | [Day 1](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2001/Readme.md) |


## Goals:
- Understand CUDA fundamentals.
- Write optimized and efficient GPU kernels.
- Explore memory hierarchy, warp scheduling, and Tensor Cores.
- Apply CUDA to deep learning and high-performance computing.


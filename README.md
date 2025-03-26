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
| 28 |  [CuTLASS](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2028)  | Tried CUTLASS , added CudaEvents and modified basic code to support like previous days naive matmul . Also made profile report using ncu |
| 29 |  [Shared Matmul Competitive](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2029)  | Wrote shared Matmul for competitive coding, optimizing performance with tiling and CUDA streams. |
| 30 |  [Vectorized Tiled Matmul](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2030)  | vectorized tiled shared mem matmul. Improved my previous days Naive matmul GFLOPS from ~ 450 to 1500 on tensara website. |
| 31 |  [Faster Float2 Vectorization](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2031)  | float2 Vectorization for faster memory coalescing |
| 32 |  [FP16 Vector Addition ](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2032)  | Optimized FP16 Vector Addition using half2 for better memory efficiency |
| 33 |  [Competitive Float2 Vector Addition ](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2033)  | Optimized CUDA kernel for element-wise vector addition using float2 for memory coalescing and efficiency. |
| 34 |  [Cmake SGEMM](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2034)  |Implemented SGEMM on square matrices, tested on RTX 3060 (CMake) and Nvidia T4 (test website). Optimizations include shared memory tiling, float2 vectorized operations, and efficient memory access for better GPU performance. |
| 35 |  [ReLU](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2035)  | Simple ReLU (Rectified Linear Unit) activation function in CUDA |
| 36 |  [Leaky ReLU](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2036)  |Leaky ReLU (Leaky Rectified Linear Unit) activation function in CUDA |
| 37 |  [Alphatensor](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2037)  |Deployed Google Deepmind Alphatensor matmul locally in my 3060. |
| 38 |  [Basic PTX](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2038)  |Learned about running PTX code and its advantages in various metrics thoroughly also analyzed compiler-generated PTX to but struggled with installation will complete this tomorrow. |
| 39 |  [Inline PTX](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2039)  |More PTX testing , command lines , cubin , Had Cuda API errors : probably bad installation . Used Compiler explorer website explore more ptx stuf and compiling.Locally also tested inline PTX assembly to load integers from global memory, add a constant value, and store the results back in global memory. |
| 40 |  [More Inline PTX](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2040)  |Did multiple inline PTX cuda snippets/functionality separately like popc , Membar , rcp and shufl .Struggle with errors on both locally and compiler explorer with only shufl type so switched to cuda intrinsic __shuffle_sync() |
| 41 |  [MLIR - 1](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2041)  |Worked on integrating MLIR with CUDA and successfully executed matrix addition.Initially faced issues with the gpu.launch method throwing numerous errors that even GPT couldn't resolve. Dropped that approach and directly integrated with the CUDA runtime. |
| 42 |  [MLIR - 2](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2042)  |More MLIR stuff, The installation time and and Figuring out deprecated commands like from cpu-runner --> runner  |
| 43 |  [INT8 Matmul](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2043)  |Wrote INT8 Matmul and compared it with FP32. The INT8 version was faster, but I messed up scaling and stream usage, causing high errors. Will try to reduce this MAE |
| 44 |  [cuSolver - 1 ](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2044)  |Also busy with college assignments pushed one code a day . Solved Linear System using cuSolver (LU Decomposition)|
| 45 |  [cuSolver - 2 ](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2045)  |QR Factorization |
| 46 |  [cuSolver - 3 ](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2046)  |Cholesky Decomposition and Eigenvalue & Eigenvector|
| 47 |  [cuSolver - 4 ](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2047)  |Singualr Value Decomposition|
| 48 |  [cuSPARSE - 1 ](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2048)  |Sparse Matmul with cuSPARSE[CSR]|
| 49 |  [cuSPARSE - 2 ](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2049)  |Compression with Grids of cuSolver [Dense]vs cuSPARSE [Sparse]|
| 50 |  [Q-learning - 1](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2050)  |Used Q-learning with sparse matrices (CSR format) to make it efficient.|
| 51 |  [Q-learning - 2](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2051)  |Started learning Tabular RL since I already tried previous day Q learning so today I Improved how the agent explores using Boltzmann exploration and Epsilon-Greedy.|
| 52 |  [Multi-Armed Bandits - 1](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2052)  |Moved from Q-learning to Multi-Armed Bandits to learn action selection strategies.|
| 53 |  [Markov Decision Process](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2053)  |Working With MDP sim where it learns from rewards from basic grid .|
| 54 |  [Q-Learning -3](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2054)  |Q-learning algorithm , achieved Unstable Q-values result , will improve.|
| 55 |  [Q-Learning -4](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2055)  |Q-values are more controlled Almost 50, making more stable agent.The extreme values (99.999, 88.999) from Day 54 are gone.|
| 56 |  [SARSA](https://github.com/ShlokVFX/100-days-cuda/blob/main/Day%2056)  |Implemented SARSA today. It is much more stable and less aggressive than Q-learning, as it follows the current policy instead of always taking the greedy action.|



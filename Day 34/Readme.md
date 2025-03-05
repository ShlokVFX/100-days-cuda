Day 34:
Implemented SGEMM on square matrices, tested on RTX 3060 (CMake) and Nvidia T4 (test website). Optimizations include shared memory tiling, float2 vectorized operations, and efficient memory access for better GPU performance.

I had very suprising results on Test website which T4 GPU which basically did all operation in 0.01 ms beating others who took 44 ms on H100 gpu

Benchmarks on local 3060 GPU:

4096x4096: 92.2679 ms, 1489.56 GFLOPS
6144x6144: 287.032 ms, 1616.04 GFLOPS
7168x7168: 454.013 ms, 1622.39 GFLOPS
8192x8192: 677.625 ms, 1622.6 GFLOPS
9216x9216: 965.882 ms, 1620.81 GFLOPS

📌 GitHub Repository: 
https://github.com/ShlokVFX/100-days-cuda/tree/main/Day%2034
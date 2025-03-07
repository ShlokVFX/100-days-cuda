Day 37:
Deployed Google Deepmind Alphatensor matmul locally in my 3060.
Initial results succeeded:

Multiplying 8192 x 8192 matrices
========================================
Strassen^2 vs `jnp.dot`: 1.20% speedup
AlphaTensor GPU-optimized vs `jnp.dot`: 2.38% speedup
AlphaTensor TPU-optimized vs `jnp.dot`: 0.61% speedup

Multiplying 14336 x 14336 matrices
========================================
Strassen^2 vs `jnp.dot`: 19.02% speedup
AlphaTensor GPU-optimized vs `jnp.dot`: 20.14% speedup
AlphaTensor TPU-optimized vs `jnp.dot`: 18.77% speedup


It failed later with 14336, 16384, 18432, 20480 matrix sizes because Out of Memory (OOM) error
Also tried implementing algorithm in Cuda.

📌 GitHub Repository: 
https://github.com/ShlokVFX/100-days-cuda/tree/main/Day%2037


```markdown
# AlphaTensor GPU Benchmarking in CUDA

This repository contains a CUDA implementation of an AlphaTensor-inspired matrix multiplication benchmark. The code compares a naive GEMM baseline with an “AlphaTensor GPU-optimized” algorithm for large matrix multiplications. This demonstration is based on a simplified bilinear factorization for 3×3 matrix multiplication similar in spirit to the factorization discovered by Google DeepMind's AlphaTensor.

## Overview

The benchmark performs multiplication on large matrices (e.g., 8192×8192) by partitioning them into 3×3 blocks. Two approaches are implemented:

- **Naive GEMM Kernel:**  
  A baseline implementation that performs standard matrix multiplication in a tiled fashion.

- **AlphaTensor-inspired Kernel:**  
  An optimized kernel that computes each 3×3 block using a bilinear factorization technique with dummy coefficients. This kernel:
  - Computes intermediate products \( p[r] \) for \( r = 0, \dots, R-1 \) using a simplified formula.
  - Combines the intermediates with a coefficient matrix \( W \) to generate the output block.

## Running the Benchmark

After compiling, run the executable:

```bash
./ATT
```
![image](https://github.com/user-attachments/assets/076041a4-9a6d-49e0-97c6-dfaadc049da1)

The program will output:
- The overall matrix dimensions.
- Execution time for the naive GEMM kernel.
- Execution time for the AlphaTensor-inspired kernel.
- The computed speedup (percentage) of the AlphaTensor kernel versus the baseline.

## Limitations and Future Work

- **Simplified Factorization:**  
  The current kernel uses only the first column of each 3×3 block to compute intermediate products. A full implementation would involve a linear combination of all elements in the block.

- **Dummy Coefficients:**  
  All factorization coefficients are set to 1.0 for demonstration. Integrating actual learned coefficients from the AlphaTensor output would yield meaningful results.

- **Optimization:**  
  Further tuning (e.g., enhanced memory access patterns, shared memory tiling, and better thread management) may improve performance.

## License

[Insert License Information Here]

## Acknowledgments

This project is inspired by Google DeepMind’s AlphaTensor work and serves as a demonstration of how such an algorithm can be modeled and benchmarked in CUDA.

```

---

Feel free to customize the README (e.g., license, acknowledgments) as needed for your repository.

✅ **Basics of GPU Architecture & Memory Types**  
- **Global Memory**: Accessible by all threads but has high latency and low bandwidth.
- **Shared Memory**: Faster than global memory, shared among threads in a block, and used for data sharing and synchronization.
- **Local Memory**: Memory allocated for individual threads, stored in global memory, and has high latency.
- **Registers**: The fastest memory type, allocated per thread, used for frequently accessed variables.

✅ **Writing Your First CUDA Kernel**  
- **CUDA Kernels (`__global__` functions)**: Special functions executed on the GPU and launched from the host.
- **Kernel Launch Configuration**: Specifies the number of threads and blocks.
- **Example of a Simple CUDA Kernel**:
   ```cpp
   __global__ void add(int *a, int *b, int *c) {
       int idx = threadIdx.x;
       c[idx] = a[idx] + b[idx];
   }
   ```

✅ **Thread Hierarchy**  
- **Grids and Blocks**: A kernel is launched in a grid of blocks, and each block contains multiple threads.
- **Threads**: Basic execution units within a block.
- **Warps**: A group of 32 threads executed in SIMT (Single Instruction Multiple Thread) fashion.
- **Mapping computations to threads**: Efficient mapping improves performance by optimizing memory access patterns.

✅ **CUDA Execution Model**  
- **SIMD (Single Instruction Multiple Data)**: Traditional parallelism where multiple processing elements execute the same instruction.
- **SIMT (Single Instruction Multiple Threads)**: NVIDIA's execution model where each thread has its own program counter but executes in groups of 32 (warps).
- **Warp Scheduling and Divergence**: Threads in a warp execute in lockstep; branching can cause divergence, leading to performance loss.

✅ **CUDA Profiling with NVIDIA Nsight Systems**  
- **Importance of CUDA Profiling**: Profiling helps identify bottlenecks, optimize memory access, and improve kernel performance.
- **NVIDIA Nsight Systems**: A powerful tool for analyzing CUDA applications, tracking GPU and CPU interactions, and optimizing performance.
- **Key Features of Nsight Systems**:
  - Timeline view of kernel execution and memory transfers.
  - Identification of warp divergence and inefficient memory accesses.
  - CPU-GPU synchronization analysis to reduce idle times.
- **Basic Profiling Command**:
   ```sh
   nsight-sys-cli -capture-range start:stop ./outputfile
   ```
## License
This project is licensed under the MIT License.



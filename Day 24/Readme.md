
Question: "How would you debug a CUDA kernel causing GPU exceptions (e.g., illegal address)?"

To debug a CUDA kernel causing an illegal address exception, I first run the kernel with cuda-memcheck to pinpoint the exact location of the illegal access. I then carefully review my index calculations and pointer arithmetic to ensure that no thread is accessing memory out of bounds. I add error checks immediately after kernel launches using cudaGetLastError() and cudaDeviceSynchronize() to catch any errors as early as possible. Finally, if needed, I use CUDA-GDB or Nsight tools to step through the code and inspect variable values and shared memory usage, making sure all shared memory accesses are synchronized correctly.

-------------------------------------------------------------------------------------------------------------------------------------------------

Question: "Explain how warp divergence affects occupancy and strategies to mitigate it in reduction kernels."

Warp divergence happens when threads in a warp follow different execution paths because of conditional branches, which causes some threads to sit idle while others execute. In reduction kernels, this is especially problematic because it can reduce the effective occupancy, leading to lower throughput. To mitigate this, I would use warp-level intrinsics like __shfl_down_sync for a warp-synchronous reduction, unroll loops to remove conditional branches, and structure the reduction in two stages: first a warp-level reduction, followed by a block-level reduction. These techniques ensure that most threads execute uniformly, improving occupancy and overall performance."

-------------------------------------------------------------------------------------------------------------------------------------------------

Question: "A matrix multiplication kernel’s performance drops by 40% when increasing matrix width from 1024 to 2048. Outline your analysis strategy."

When facing a 40% performance drop in a matrix multiplication kernel when increasing the matrix width from 1024 to 2048, I would first verify the kernel’s correctness and examine input data patterns. I’d use Nsight Compute to compare occupancy, register usage, and active warps per SM between the two cases. Next, I’d analyze memory access patterns and cache behavior—comparing global memory throughput, cache hit rates, and any changes in shared memory efficiency or bank conflicts. I’d also look at instruction-level metrics and stall reasons to see if the kernel becomes more memory-bound or experiences higher divergence with the larger matrix. Based on these insights, I’d adjust tiling/blocking strategies, optimize memory accesses, or refactor code to reduce register pressure, thereby aiming to recover performance on the larger matrix.

-------------------------------------------------------------------------------------------------------------------------------------------------

Question: "A kernel’s achieved occupancy is 25%. What hardware counters would you analyze to identify the bottleneck?"

Check register usage (sm__warps_active → Too high = lower registers per thread).
Check memory stalls (smsp__warp_issue_stalled_long_scoreboard → High = Optimize memory access).
Check launched warps (sm__warps_launched → Low = Increase threads per block).
Check warp divergence (smsp__thread_inst_executed_per_inst_executed → Low = Fix branching).


Summary

    Naive: Simple per–element kernel (very easy but slow).
    Tiled: Uses shared memory tiling to boost memory reuse.
    Tiled + Unrolling: Further reduces loop overhead.
    Thread Tiling: Each thread computes multiple C–elements to improve arithmetic intensity.
    WMMA/Tensor Core: Leverages NVIDIA’s Tensor Cores via the WMMA API (fastest for FP16 workloads).

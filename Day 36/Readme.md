# Leaky ReLU Activation Function in CUDA

## Overview
Leaky ReLU (Leaky Rectified Linear Unit) is an activation function commonly used in deep learning. Unlike standard ReLU, which outputs zero for negative inputs, Leaky ReLU allows a small, nonzero gradient for negative values to prevent neuron deadlocks.

This CUDA implementation is optimized for parallel execution on GPUs, allowing efficient batch processing of activations.

## Mathematical Definition
The Leaky ReLU function is defined as:

\[
 f(x) = \begin{cases} 
     x & x > 0 \\
     \alpha x & x \leq 0
 \end{cases}
\]

where \( \alpha \) is a small slope factor (typically 0.01) for negative inputs.

## Implementation Details
- Utilizes CUDA kernels to apply Leaky ReLU across a large number of input values in parallel.
- Supports both in-place and out-of-place computation.
- Uses grid-stride loops for better memory access patterns and scalability.
- Optimizations may include shared memory and warp-level operations for improved performance.

## Prerequisites
Ensure you have:
- An NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- A working compilation setup with `nvcc`

## Compilation and Execution
To compile the CUDA program:
```sh
nvcc -o leaky_relu leaky_relu.cu
```
To run the executable:
```sh
./leaky_relu
```

## Sample CUDA Kernel
```cpp
__global__ void leakyReLU(float* input, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = (input[idx] > 0) ? input[idx] : alpha * input[idx];
    }
}
```

## Performance Optimizations
- **CUDA Streams**: Overlapping computation for improved performance.
- **Shared Memory**: Reducing global memory accesses for frequently used data.
- **Vectorized Loads**: Using memory coalescing techniques for efficient access.

## Future Enhancements
- Integration with deep learning frameworks like TensorFlow and PyTorch.
- Extended support for half-precision (`FP16`) for performance gains on modern GPUs.
- Further tuning using CUDA Graphs and occupancy-based optimizations.

## References
- **Deep Learning** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville.
- NVIDIA CUDA Programming Guide.

## License
This implementation is released under the MIT License.
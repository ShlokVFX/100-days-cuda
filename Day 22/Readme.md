# Triton Fused Softmax

## Overview

This project demonstrates the implementation of a fused softmax operation using Triton, a language and compiler for writing custom deep learning primitives. Fused softmax is an optimized version of the softmax function that combines multiple operations into a single kernel, improving performance by reducing memory bandwidth and increasing computational efficiency.

## Implementation Details

The implementation leverages Triton's ability to write highly efficient GPU kernels. The fused softmax operation is designed to:

1. Load input data from global memory.
2. Compute the maximum value for numerical stability.
3. Subtract the maximum value from each element to prevent overflow.
4. Exponentiate the adjusted values.
5. Compute the sum of the exponentiated values.
6. Divide each exponentiated value by the sum to obtain the final softmax probabilities.

## Expected Output

The expected output of the Triton fused softmax operation is a tensor of the same shape as the input, where each element represents the softmax probability. The sum of the probabilities along the specified dimension should be equal to 1.

For example, given an input tensor:

```
[[1.0, 2.0, 3.0],
 [1.0, 2.0, 3.0]]
```

The expected output would be:

```
[[0.09003057, 0.24472847, 0.66524096],
 [0.09003057, 0.24472847, 0.66524096]]
```

Each row sums to 1, demonstrating the properties of the softmax function.

## Usage

To use the Triton fused softmax implementation, follow these steps:

1. Install Triton by following the instructions on the [Triton GitHub repository](https://github.com/openai/triton).
2. Clone this repository and navigate to the project directory.
3. Run the provided script to execute the fused softmax operation on sample data.

```bash
git clone <repository_url>
cd <repository_directory>
python fused_softmax.py
```

## Conclusion

The Triton fused softmax implementation provides a highly efficient way to compute the softmax function on GPUs. By fusing multiple operations into a single kernel, it achieves better performance compared to traditional implementations. This project serves as a practical example of leveraging Triton for custom deep learning primitives.

# Triton Basics

This repository contains examples and explanations of using Triton for various operations such as vector operations, fused softmax, and matrix multiplication.

## Table of Contents
- [Introduction](#introduction)
- [Vector Operations](#vector-operations)
- [Fused Softmax](#fused-softmax)
- [Matrix Multiplication](#matrix-multiplication)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Triton is a language and compiler for writing highly efficient GPU code. This repository provides basic examples to help you get started with Triton.

## Vector Operations
Vector operations are fundamental in many computational tasks. This section covers basic vector operations using Triton.

### Example
```python
import triton
import triton.language as tl

@triton.jit
def vector_addition(a, b, c, N):
    pid = tl.program_id(0)
    idx = pid * 256 + tl.arange(0, 256)
    mask = idx < N
    a = tl.load(a + idx, mask=mask)
    b = tl.load(b + idx, mask=mask)
    c = a + b
    tl.store(c + idx, c, mask=mask)
```

## Fused Softmax
Fused softmax is an optimized version of the softmax function that combines multiple operations into a single kernel.

### Example
```python
import triton
import triton.language as tl

@triton.jit
def fused_softmax(x, y, N):
    pid = tl.program_id(0)
    idx = pid * 256 + tl.arange(0, 256)
    mask = idx < N
    x = tl.load(x + idx, mask=mask)
    max_x = tl.max(x, axis=0)
    exp_x = tl.exp(x - max_x)
    sum_exp_x = tl.sum(exp_x, axis=0)
    y = exp_x / sum_exp_x
    tl.store(y + idx, y, mask=mask)
```

## Matrix Multiplication
Matrix multiplication is a common operation in many applications. This section demonstrates how to perform matrix multiplication using Triton.

### Example
```python
import triton
import triton.language as tl

@triton.jit
def matmul(a, b, c, M, N, K):
    pid = tl.program_id(0)
    row = pid // N
    col = pid % N
    acc = 0.0
    for k in range(K):
        acc += tl.load(a + row * K + k) * tl.load(b + k * N + col)
    tl.store(c + row * N + col, acc)
```

## Getting Started
To get started with Triton, follow these steps:
1. Install Triton: `pip install triton`
2. Clone this repository: `git clone https://github.com/yourusername/triton-basics.git`
3. Navigate to the project directory: `cd triton-basics`
4. Run the examples: `python examples/vector_addition.py`

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
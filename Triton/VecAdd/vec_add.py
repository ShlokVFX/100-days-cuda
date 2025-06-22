import time
import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------------------
# Triton Kernel for Vector Addition
# ------------------------------------------------------------------------------

@triton.jit
def vector_add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    BLOCK_SIZE:tl.constexpr
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    mask = offset < n_elements
    a = tl.load(a_ptr + offset , mask=mask)
    b = tl.load(b_ptr + offset , mask=mask)
    c = a + b
    tl.store(c_ptr + offset, c ,mask=mask)

def vector_add_triton(A: torch.Tensor, B: torch.Tensor)-> torch.Tensor:
    n_elements = A.numel()
    C = torch.empty_like(A)
    grid = lambda meta: (triton.cdiv(n_elements,meta['BLOCK_SIZE']),)
    vector_add_kernel[grid](A,B,C,n_elements,BLOCK_SIZE=1024)
    return C

# ------------------------------------------------------------------------------
# Benchmarking Function
# ------------------------------------------------------------------------------

def benchmark(func, *args, n_warmup=10, n_iters=100):
    """
    Benchmarks a function by running warm-up iterations followed by timed iterations.
    
    Args:
        func (callable): The function to benchmark.
        *args: Arguments to pass to the function.
        n_warmup (int): Number of warm-up iterations (to exclude startup overhead).
        n_iters (int): Number of iterations for timing.
    
    Returns:
        float: Average execution time per iteration in milliseconds.
    """
    # Warm-up runs to ensure any one-time setup is complete (e.g. CUDA context)
    for _ in range(n_warmup):
        func(*args)
    torch.cuda.synchronize()  # Ensure warm-up kernels have finished

    # Start timing
    start = time.perf_counter()
    for _ in range(n_iters):
        func(*args)
    torch.cuda.synchronize()  # Wait for all GPU operations to finish
    end = time.perf_counter()

    # Calculate the average execution time (in milliseconds)
    avg_time_ms = (end - start) / n_iters * 1000
    return avg_time_ms

# ------------------------------------------------------------------------------
# Main: Compare and Benchmark Triton Kernel vs. PyTorch Implementation
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    # Create two example vectors on the GPU (stress test with a large number of elements)
    n = 1024 * 1024 * 10  # e.g., 10 million elements
    A = torch.arange(0, n, device='cuda', dtype=torch.float32)
    B = torch.arange(n, 2 * n, device='cuda', dtype=torch.float32)

    # Validate correctness by comparing results from Triton and PyTorch
    C_triton = vector_add_triton(A, B)
    C_pytorch = A + B

    if torch.allclose(C_triton, C_pytorch):
        print("Success: The Triton result matches the PyTorch result!")
    else:
        print("Error: The results do not match.")

    # Benchmark the Triton kernel
    triton_time = benchmark(vector_add_triton, A, B, n_warmup=10, n_iters=100)
    print(f"Average execution time (Triton): {triton_time:.3f} ms")

    # Benchmark the PyTorch implementation
    def pytorch_add(A, B):
        return A + B

    pytorch_time = benchmark(pytorch_add, A, B, n_warmup=10, n_iters=100)
    print(f"Average execution time (PyTorch): {pytorch_time:.3f} ms")
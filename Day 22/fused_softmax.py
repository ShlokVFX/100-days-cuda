import torch
import triton
import triton.language as tl

# Check if CUDA is available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Naive Softmax in PyTorch
def naive_softmax(x):
    x_max = x.max(dim=1, keepdim=True)[0]  # Keep dimensions
    z = x - x_max
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1, keepdim=True)
    return numerator / denominator

# Triton Softmax Kernel
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in range(row_start, n_rows, row_step):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols

        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

# Triton-based Softmax function
def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 2  # Adjusted for memory usage

    y = torch.empty_like(x)

    # Set num_programs based on row count
    num_programs = min(1024, n_rows)  # Limiting to 1024 threads for safety

    # ✅ Fix: Change grid to (num_programs, 1) instead of (num_programs,)
    softmax_kernel[(num_programs, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)

    return y

# Test correctness
torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
y_triton = softmax(x)
y_torch = torch.softmax(x, dim=1)

assert torch.allclose(y_triton, y_torch, atol=1e-6), "Triton softmax does not match PyTorch softmax"

print("✅ Softmax implementation is correct!")

# Benchmarking
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'], 
        x_vals=[128 * i for i in range(2, 100)], 
        line_arg='provider',  
        line_vals=['triton', 'torch'],
        line_names=["Triton", "Torch"],  
        styles=[('blue', '-'), ('green', '-')],  
        ylabel="GB/s",  
        plot_name="softmax-performance",  
        args={'M': 4096}, 
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, dim=-1))
    elif provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    
    gbps = (2 * x.numel() * x.element_size() * 1e-9) / (ms * 1e-3)
    return gbps

benchmark.run(show_plots=True, print_data=True)

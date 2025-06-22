import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

def naive_softmax(x):
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(1)
    out = numerator / denominator[:, None]
    return out

@triton.jit
def _softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_row_ptr = input_ptr + row_id * input_row_stride + col_offsets
    mask = col_offsets < n_cols

    row = tl.load(input_row_ptr, mask=mask, other=float('-inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_row_ptr = output_ptr + row_id * output_row_stride + col_offsets
    tl.store(output_row_ptr, softmax_output, mask=mask)

def softmax(x):
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)  # Triton does not support very large blocks

    y = torch.empty_like(x)

    grid = lambda META: (n_rows,)
    _softmax_kernel[grid](
        x, y,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return y

def test_softmax_kernel(size: tuple, atol=1e-3, rtol=1e-3, device=DEVICE):
    assert type(size) is tuple and len(size) == 2
    torch.manual_seed(0)
    x = torch.randn(size[0], size[1], device=device)
    z_tri = softmax(x)
    z_ref = torch.softmax(x, dim=1)
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("passed")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 20)],  # reduced from 2..100
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='softmax-perf',
        args={'M': 1024},  # reduced from 4096
    )
)

def benchmark(M, N, provider):

    x = torch.rand((M, N), device=DEVICE, dtype=torch.float32)

    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)


    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x,axis=1))
    elif provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

if __name__ == "__main__":
    test_softmax_kernel(size=(1823, 781))

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=True)

#!POPCORN leaderboard amd-fp8-mm

# This is a submission template for popcorn leaderboard 'amd-fp8-mm'.
# Your task is as follows:
# > 
# > You will implement a custom fp8-blockwise matmul kernel optimized for MI300.
# > You will be given single-precision scaling factors for your matrices.
# > The shapes of all outer and inner dimensions of tensors are from DeepSeek-R1.
# > To be explicit, you will be given a tuple of tensors:
# > ```
# > (a, b, a_scale, b_scale, c)
# > ```
# > where `a` and `b` are the input matrices, `a_scale` and `b_scale` are the scaling factors for `a` and `b` respectively,
# > and `c` is the output matrix:
# > * `a` is M x K in column-major order in e4m3fnuz
# > * `b` is N x K in column-major order in e4m3fnuz
# > * `a_scale` is M x K // 128 in column-major order in fp32
# > * `b_scale` is N // 128 x K // 128 in column-major order in fp32
# > * `c` is M x N in ROW-major order in bf16
# > 
# > Matrix sizes `m` and `n` are divisible by 64, `k` is divisible by 128.
# > 
# > The ranking criteria is the geometric mean of the benchmark results.
# > 
# > For the grand price, your kernel will be evaluated against the speed of light analysis
# > and the solution closest to the speed of light will be awarded the grand price.
# > ```
# > The speed of light analysis is:
# >  M       N       K     time[us]
# > 1024    1536    7168      8.63
# > 1024    4608    7168     25.89
# > 6144    1536    7168     51.78
# > 6144    4608    7168    155.30
# > 1024    7168     256      3.17
# > 6144    7168     256     17.27
# > ```
# The deadline for this leaderboard is 2025-09-02 00:00:00+00:00

# You can automatically route this file to specific GPUs by adding a line
# `#!POPCORN gpus <GPUs>` to the header of this file.
# Happy hacking!

from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of block-scale fp8 gemm
    Args:
        data: Tuple that expands to:
            a: torch.Tensor[float8_e4m3fnuz] of shape [m, k],
            b: torch.Tensor[float8_e4m3fnuz] of shape [n, k],
            a_scale: torch.Tensor[float32] of shape [m, k // 128],
            b_scale: torch.Tensor[float32] of shape [n // 128, k // 128],
            c: torch.Tensor[bfloat16] of shape [m, n]
    Returns:
        Tensor containing output in bf16
    """
    # c: [m, n] is pre-allocated memory to avoid timing allocation overhead.
    a, b, a_scale, b_scale, c = data

    # Your implementation here

    return c


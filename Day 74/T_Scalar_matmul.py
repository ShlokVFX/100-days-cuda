import triton
import triton.language as tl

def solution(input_matrix, scalar, output_matrix, n: int):
    BLOCK_M = 128
    BLOCK_N = 128

    @triton.jit
    def scalar_mul_kernel(input_ptr, scalar_val, output_ptr, n,
                          stride_in, stride_out,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask_m = offs_m < n
        mask_n = offs_n < n

        # 2D tile of pointers
        in_ptrs = input_ptr + offs_m[:, None] * stride_in + offs_n[None, :]
        out_ptrs = output_ptr + offs_m[:, None] * stride_out + offs_n[None, :]

        # Vectorized load -> multiply -> store
        a = tl.load(in_ptrs, mask=mask_m[:, None] & mask_n[None, :])
        tl.store(out_ptrs, a * scalar_val, mask=mask_m[:, None] & mask_n[None, :])

    grid = lambda META: (triton.cdiv(n, META['BLOCK_M']),
                         triton.cdiv(n, META['BLOCK_N']))

    scalar_mul_kernel[grid](
        input_matrix, scalar, output_matrix, n,
        stride_in=n, stride_out=n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )

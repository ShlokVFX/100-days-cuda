import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

def solution(d_input1, d_input2, d_output, n: int):
    BLOCK_SIZE = 1024
    grid = lambda META: (triton.cdiv(n, BLOCK_SIZE),)
    vector_add_kernel[grid](
        d_input1, d_input2, d_output,
        n_elements=n,
        BLOCK_SIZE=BLOCK_SIZE
    )

import triton
import triton.language as tl

def solution(input_a, input_b, output_c, m: int, k: int):
    BLOCK_SIZE = 128

    @triton.jit
    def matvec_kernel(a_ptr, b_ptr, c_ptr, M, K, 
                      stride_am, stride_ak, stride_bk, stride_cm,
                      BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        row = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = row < M

        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        for offset_k in range(0, K, BLOCK_SIZE):
            col = offset_k + tl.arange(0, BLOCK_SIZE)
            col_mask = col < K

            a_tile = tl.load(a_ptr + row[:, None] * stride_am + col[None, :] * stride_ak,
                             mask=mask[:, None] & col_mask[None, :],
                             other=0.0)
            b_tile = tl.load(b_ptr + col * stride_bk, mask=col_mask, other=0.0)

            acc += tl.sum(a_tile * b_tile[None, :], axis=1)

        tl.store(c_ptr + row * stride_cm, acc, mask=mask)

    grid = lambda META: (triton.cdiv(m, META['BLOCK_SIZE']),)
    
    matvec_kernel[grid](
        input_a, input_b, output_c, m, k,
        stride_am=k, stride_ak=1,
        stride_bk=1, stride_cm=1,
        BLOCK_SIZE=BLOCK_SIZE
    )

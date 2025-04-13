import triton
import triton.language as tl

def solution(input_a, input_b, output_c, m: int, n: int, k: int):
    @triton.jit
    def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K):
        row = tl.program_id(0)
        col = tl.program_id(1)
        
        if row >= M or col >= N:
            return
            
        acc = 0.0
        for i in range(K):
            a_val = tl.load(a_ptr + row * K + i)
            b_val = tl.load(b_ptr + i * N + col)
            acc += a_val * b_val
            
        tl.store(c_ptr + row * N + col, acc)

    grid = (m, n)
    matmul_kernel[grid](input_a, input_b, output_c, m, n, k)
    
    return output_c
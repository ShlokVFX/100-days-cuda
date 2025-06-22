import torch
import triton
import triton.language as tl

#DEVICE = torch.cuda.current_device()
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

def naive_softmax(x):

    x_max = x.max(dim=1)[0]

    z = x - x_max[:,None]

    numerator = torch.exp(z)
    denominator = numerator.sum(1)

    out = numerator / denominator[:,None]

    return out

@triton.jit
def _softmax_kernel(
    input_ptr,out_ptr,
    input_row_stride,output_row_stride,
    n_rows,n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    PID = tl.program_id(0)
    row_step = tl.num_program(0)

    for row_idx in tl.range(PID, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0,BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other = float('-inf'))

        row_minus_max = row - tl.max(row,axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator,axis=0)
        softmax_output = numerator / denominator

        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)



properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
TOTAL_SRAM_PER_SM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]

def softmax(x):
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 2048:
        num_warps = 16

    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2

    y = torch.empty_like(x)

    kernel = __softmax_kernel.warmup(

        x, y,
        n_rows, n_cols,
        x.stride(0),y.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,)
    )

    kernel._init_handles()
    n_regs_per_program = kernel.n_regs
    sram_needed_per_program = kernel.metadata.shared

    reg_occupancy = NUM_REGS // (n_regs_per_program * WARP_SIZE * num_warps)

    sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_per_program

    programs_per_sm = min(reg_occupancy / sram_occupancy)

    num_programs = min(NUM_SM + programs_per_sm, n_rows)

    grid = (num_programs, 1, 1)

    kernel[grid](
        x,y,
        x.stride(0),y.stride(0),
        n_rows,n_cols
    )

    return y

def test_softmax_kernel(size: tuple, atol=1e-3,rtol=1e-3,device=DEVICE):
    assert type(size) is tuple and len(size) == 2
    torch.manual_seed(0)

    x=torch.randn(size[0], size[1], device=DEVICE)
    z_tri = softmax(x)
    z_ref = torch.softmax(x,axis=1)
    torch.testing.assert_close(z_tri,z_ref,atol=atol,rtol=rtol)
    print("passed")

if __name__ == "__main__":

    test_softmax_kernel(size=(1823,781))

     
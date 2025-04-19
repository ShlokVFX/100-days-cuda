#!POPCORN leaderboard amd-fp8-mm
from task import input_t, output_t
import torch
from torch.utils.cpp_extension import load_inline
import time
import os
import sys

if "PYTORCH_ROCM_ARCH" not in os.environ:
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx942:xnack-"

kernel_cpp = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <pybind11/pybind11.h>
#include <iostream>

#define HIP_CHECK(condition)                                                                  @
    do                                                                                        @
    {                                                                                         @
        hipError_t error = condition;                                                         @
        if(error != hipSuccess)                                                               @
        {                                                                                     @
            std::cout << "Error " << hipGetErrorName(error) << '(' << error << ')' << ": "    @
                      << hipGetErrorString(error) << " in " << __func__ << " at " << __FILE__ @
                      << ':' << __LINE__ << '@n';                                             @
            exit(error);                                                                      @
        }                                                                                     @
    }                                                                                         @
    while(false)

using fp8 = __hip_fp8_e4m3_fnuz;
using bf16 = __hip_bfloat16;
using fp32 = float;

template <typename T, typename U>
__host__ __device__
constexpr auto ceil_div(T a, U b) {
    return (a + b - 1) / b;
}

__global__ void kernel(
    const fp8* a,
    const fp8* b,
    const fp32* as,
    const fp32* bs,
    bf16* c,
    int m,
    int n,
    int k
) {
    int i_m = blockIdx.x * blockDim.x + threadIdx.x;
    int i_n = blockIdx.y * blockDim.y + threadIdx.y;

    int sn = ceil_div(n, 128);

    fp32 result = 0;
    for (int i = 0; i < k; i += 128) {
        fp32 block_result = 0;
        #pragma loop unroll
        for (int ii = 0; ii < 128; ++ii) {
            block_result +=
                fp32(a[(i + ii) * m + i_m]) *
                fp32(b[(i + ii) * n + i_n]);
        }
        result += block_result * as[i / 128 * m + i_m] * bs[i / 128 * sn + i_n / 128];
    }

    c[i_m * n + i_n] = bf16(result);
}

void run(
    uintptr_t a_ptr,
    uintptr_t b_ptr,
    uintptr_t as_ptr,
    uintptr_t bs_ptr,
    uintptr_t c_ptr,
    int m,
    int n,
    int k
) {
    const auto* a = reinterpret_cast<const fp8*>(a_ptr);
    const auto* b = reinterpret_cast<const fp8*>(b_ptr);
    const auto* as = reinterpret_cast<const fp32*>(as_ptr);
    const auto* bs = reinterpret_cast<const fp32*>(bs_ptr);
    auto* c = reinterpret_cast<bf16*>(c_ptr);

    int block_m = 16;
    int block_n = 16;

    assert(m % block_m == 0);
    assert(n % block_n == 0);

    int grid_m = m / block_m;
    int grid_n = n / block_n;

    kernel<<<dim3(grid_m, grid_n), dim3(block_m, block_n), 0>>>(
        a,
        b,
        as,
        bs,
        c,
        m,
        n,
        k
    );
    HIP_CHECK(hipGetLastError());
}

PYBIND11_MODULE(fp8, m) {
  m.def("fp8", &run, "HIP kernel");
}
"""

hip_module = load_inline(
    name="fp8",
    cpp_sources="",
    cuda_sources=kernel_cpp.replace('@', chr(92)),
    with_cuda=True,
    verbose=False,
    extra_cuda_cflags=["-std=c++20"],
    no_implicit_headers=True,
)

first = True

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

    global first
    if first:
        print("executing on", torch.cuda.get_device_name(), file=sys.stderr)
        first = False

    a, b, a_scale, b_scale, c = data

    m, n = c.shape
    k = a.shape[1]

    hip_module.fp8(
        a.data_ptr(),
        b.data_ptr(),
        a_scale.data_ptr(),
        b_scale.data_ptr(),
        c.data_ptr(),
        m,
        n,
        k,
    )

    return c

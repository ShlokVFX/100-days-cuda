import os
import sys
import math
import numpy as np
import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
import torch.nn.functional as F

os.environ["CXX"] = "clang++"
os.environ["PYTORCH_ROCM_ARCH"] = "gfx942"

with open('./fa2.hip', 'r') as f:
    s = f.read()

cuda_src = s[:s.find('int main()')]

cuda_src += r"""

at::Tensor decoding_flash_attention2(at::Tensor Q, at::Tensor K, at::Tensor V) {
  assert(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous());
  assert(Q.sizes().size() == 4 && K.sizes().size() == 4 &&
         V.sizes().size() == 4 && Q.size(3) == 192 && K.size(3) == 192 &&
         V.size(3) == 128);

  uint32_t B = K.size(0), nh = K.size(1), N = K.size(2);

  const uint32_t dqk = 192;
  const uint32_t dv = 128;
  const uint32_t Bc = 64;

  // using namespace std;
  // cout << B << endl;
  // cout << nh << endl;
  // cout << N << endl;
  // cout << dqk << endl;
  // cout << K.index({100, 0, 0, 0}).item<float>() << endl;
  // auto K_cpu = K.contiguous().to(torch::kCPU);
  // auto x = K_cpu.data_ptr<c10::BFloat16>()[100 * 50356224];
  // cout << "nigga => " << (float) x << endl;

  at::Tensor O = at::empty({B, nh, 1, dv}, Q.options());

  const uint32_t numThreads = Bc;
  dim3 numBlocks(B * nh);
  decoding_flash_attention2_kernel<Bc, numThreads, dqk, dv>
      <<<numBlocks, numThreads>>>(
          (bf16_t *)Q.data_ptr(), (bf16_t *)K.data_ptr(),
          (bf16_t *)V.data_ptr(), (bf16_t *)O.data_ptr(), B, nh, N);

  return O;
}
"""

cpp_src = r"""
at::Tensor decoding_flash_attention2(at::Tensor Q, at::Tensor K, at::Tensor V);
"""

if sys.stdout is None:
    sys.stdout = open("/dev/stdout", "w")
if sys.stderr is None:
    sys.stderr = open("/dev/stderr", "w")

module = load_inline(
    name="decoding_flash_attention2",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["decoding_flash_attention2"],
    verbose=True,
    extra_cuda_cflags=[
        "-Ofast",
        "--offload-arch=gfx942",
        "-std=c++20",
        "-ffp-contract=fast",
        "-lhip_hcc",
         "-mcumode",
    ],
)

B = 128
nh = 128
N = 6144 + 1
dqk = 192
dv = 128

torch.manual_seed(42)

q = torch.randn(B, nh, 1, dqk, dtype=torch.bfloat16, device='cuda').contiguous()
k = torch.randn(B, nh, N, dqk, dtype=torch.bfloat16, device='cuda').contiguous()
v = torch.randn(B, nh, N, dv, dtype=torch.bfloat16, device='cuda').contiguous()

# print(k.float()[85, 37, 688, 63])
# print(k.float()[100, 0, 0, 0])

print('================================== running ==================================')

import torch.utils.benchmark as benchmark
def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6


# t = benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, q, k, v)
# print(f"The torch's spda implementation runs in {t:.3f} microseconds")

t = benchmark_torch_function_in_microseconds(module.decoding_flash_attention2, q, k, v)
print(f"The custom implementation runs in {t:.3f} microseconds")

# pred = module.decoding_flash_attention2(q, k, v)
# print(pred)

# z = ((q @ k.transpose(-1, -2))[0, 0, 0] / math.sqrt(dqk))
# print(z[:64])
# print(z[64*4:64*5])
# for i in range(64, 64 + 64, 64):
#     y = z[:i]
#     # print(y.max(), (y - y.max()).exp().sum())
#     print(F.softmax(y, dim=-1))
# print(z.max(), (z - z.max()).exp().sum())
# print(((y - y.max()).exp()).sum())
# print(((y - y.max()).exp()) @ v[0, 0].float())
# y = z[:N-1]
# print(((y - y.max()).exp()) @ v[0, 0, :N-1].float())
# y = z[:N]
# print(((y - y.max()).exp()) @ v[0, 0, :N].float())

# z = ((q.float()[-1, -1] @ k.float()[-1, -1].transpose(-1, -2))[0])
# print(k[-1,-1,0,0])
# print(k.shape)
# print(k.stride())
# print(k.float().contiguous()[:, 0, 0, 0])
# print(k.flatten()[100 * 128 * N * dqk])
# print(z[:64])
# print(z[64*4:64*5])
# for i in range(64, 1024 + 64, 64):
#     y = z[:i]
#     print(y.max(), (y - y.max()).exp().sum())
# print(z.max(), (z - z.max()).exp().sum())
# print(((y - y.max()).exp()).sum())
# print(((y - y.max()).exp()) @ v[0, 0].float())
# y = z[:N-1]
# print(((y - y.max()).exp()) @ v[0, 0, :N-1].float())
# y = z[:N]
# print(((y - y.max()).exp()) @ v[0, 0, :N].float())

# a = ((F.softmax((q.float() @ k.float().transpose(-1,-2)) / math.sqrt(k.shape[-1]), dim=-1) @ v.float())).to(torch.bfloat16)
# t = F.scaled_dot_product_attention(q, k, v)[0, 0, 0]

# y2 = ((F.softmax((q.float() @ k.float().transpose(-1,-2)) / math.sqrt(k.shape[-1]), dim=-1) @ v.float())).to(torch.bfloat16)
# y = (F.softmax((q @ k.transpose(-1,-2)) / math.sqrt(k.shape[-1]), dim=-1) @ v)
# a = y[1, 96, 0]
# b = pred[1, 96, 0]
# c = y2[1, 96, 0]
# # print(v[0,0,-1,:2])

# print(torch.allclose(y, pred, atol=5e-3))
# print((y - pred).abs().max())
# print((y - pred).abs().argmax())
# print(y[1, 96, 0, 73])
# print(pred[1, 96, 0, 73])
# # print(t)
# print(a)
# print(b)
# print(c)
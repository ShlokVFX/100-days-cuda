import math
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from task import input_t, output_t
from utils import make_match_reference
from submission import custom_kernel
from reference import generate_input

# Generate test input
batchsize = 128
dim = 7168 
dq = 1536
prefill = 6144
seed = 5291

# Create model and inputs
config, x, kv_cache = generate_input(batchsize, dim, dq, prefill, seed)

from torch.profiler import profile, record_function, ProfilerActivity

activities = [ProfilerActivity.CUDA]

with profile(activities=activities, record_shapes=True) as prof:
    with record_function("model_inference"):
        custom_kernel((config, x, kv_cache))

sort_by_keyword = "cuda_time_total"
print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))
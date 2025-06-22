import math
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from task import input_t, output_t

class RoPE(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        theta = 10000 ** (-torch.arange(0, d_model//2,dtype=torch.bfloat16) / (d_model//2))
        self.register_buffer("theta", theta)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        seq_len = x.size(-2)
        d_model = x.size(-1)
        assert d_model == self.d_model
        seq_idx = torch.arange(start_pos, start_pos + seq_len, device=x.device)
        idx_theta = torch.einsum('s,d->sd', seq_idx, self.theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)
        cos = idx_theta2.cos().to(torch.bfloat16)
        sin = idx_theta2.sin().to(torch.bfloat16)
        return x * cos + self.rotate_half(x) * sin


class KVCache(nn.Module):
    def __init__(self, kv_cache_shape: tuple) -> None:
        super().__init__()
        self.register_buffer('data', torch.zeros(kv_cache_shape, dtype=torch.bfloat16, device='cuda'))
        self.seq_len = 0
        self.zero()

    def zero(self) -> None:
        self.data.zero_()
    
    def get_data(self) -> torch.Tensor:
        return self.data

    def forward(self, c_kv: torch.Tensor) -> torch.Tensor:
        assert self.seq_len + c_kv.size(1) <= self.data.size(1), "KV Cache Exceeded"

        self.data = self.data.to(c_kv.dtype)
        self.data[
            :, self.seq_len : self.seq_len + c_kv.size(1), :
        ] = c_kv
        self.seq_len += c_kv.size(1)

        return self.data[:, :self.seq_len], self.seq_len


def tensor4x3(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = b.size(-2)
    input = a.flatten(start_dim=1, end_dim=2)
    output = torch.bmm(input, b.transpose(-1, -2))
    return output.view(a.size(0), a.size(1), a.size(2), n)

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b.t())
    
@dataclass
class Config:
    batch_size: int
    dim: int
    n_heads: int
    q_lora_rank: int 
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    seq_len: int
    max_seq_len: int
    kv_cache_shape: tuple
    Q_proj_down_weight: torch.Tensor
    Q_proj_up_weight: torch.Tensor
    KV_proj_down_weight: torch.Tensor
    KV_proj_up_weight: torch.Tensor
    wo_weight: torch.Tensor

class MLA(nn.Module):
    def __init__(self, config: Config, seq_len: int):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.nope_head_dim = config.qk_nope_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        # Down-projection matrices
        self.Q_proj_down_weight = config.Q_proj_down_weight
        self.Q_proj_up_weight = config.Q_proj_up_weight
        self.KV_proj_down_weight = config.KV_proj_down_weight
        self.KV_proj_up_weight = config.KV_proj_up_weight
        self.wo_weight = config.wo_weight

        # RoPE on half embeddings
        self.q_rope = RoPE(self.rope_head_dim)
        self.k_rope = RoPE(self.rope_head_dim)

        # Output projection
        self.eps = 1e-6
   
    def forward(self, x: torch.Tensor, kv_cache: KVCache) -> torch.Tensor:
        # seq_len = 1 always here
        batch_size, seq_len, model_dim = x.size()

        ################################################################################
        #                 Step 1: Handle down-projection + KV cache                    #
        ################################################################################
        q_lora = matmul(x, self.Q_proj_down_weight)
        kv_lora = matmul(x, self.KV_proj_down_weight)
        kv_lora, kv_len = kv_cache(kv_lora)
        query_pos = kv_len - 1

        ################################################################################
        #                  Step 2: Up-project and prepare NoPE + RoPE                  #
        ################################################################################

        # Handle queries Q first
        q_nope_and_rope = matmul(q_lora, self.Q_proj_up_weight).view(
            batch_size, seq_len, self.n_heads, self.nope_head_dim + self.rope_head_dim)
        q_nope, q_rope = torch.split(q_nope_and_rope, [self.nope_head_dim, self.rope_head_dim], dim=-1)

        # Compute RoPE for queries and combine with no-RoPE part
        q_rope = q_rope.permute(2, 0, 1, 3) # n_heads x bs x seq_len x rope_head_dim
        q_rope = self.q_rope(q_rope, start_pos=query_pos)

        q_nope = q_nope.permute(2, 0, 1, 3) # n_heads x bs x seq_len x rope_head_dim

        # Handle keys and values K/V. V does not need RoPE
        kv_nope, k_rope = torch.split(kv_lora, [self.kv_lora_rank, self.rope_head_dim], dim=-1)
        # Compute RoPE for keys and combine with no-RoPE part
        k_rope = self.k_rope(k_rope)
        kv_lora = torch.cat((kv_nope, k_rope), dim=-1)
                
        ################################################################################
        #                        Compute Multi-head Attention                          #
        ################################################################################

        KV_proj_up_weight = self.KV_proj_up_weight.view(self.n_heads, self.nope_head_dim + self.v_head_dim, -1)
        K_proj_up_weight, V_proj_up_weight = KV_proj_up_weight.split((self.nope_head_dim, self.v_head_dim), 1)
        K_proj_up_weight = K_proj_up_weight.transpose(-1, -2)
        kv_nope = kv_nope.transpose(-1, -2)
        scores = tensor4x3(q_nope, K_proj_up_weight)
        scores = torch.cat((scores, q_rope), dim=-1)
        scores = tensor4x3(scores.transpose(0, 1), kv_lora) * (1.0 / math.sqrt(self.rope_head_dim + self.nope_head_dim))
        attn = F.softmax(scores, dim=-1).to(torch.bfloat16)
        y = tensor4x3(attn, kv_nope)
        y = tensor4x3(y.transpose(0, 1), V_proj_up_weight)
        y = y.transpose(0, 1).reshape(batch_size, 1, -1)
        
        y = matmul(y, self.wo_weight)
        return y, kv_cache.get_data()
    

def custom_kernel(data: input_t) -> output_t:
    config, x, kv_cache = data
    model = MLA(config, kv_cache.seq_len).to('cuda')
    output, kv_cache = model(x, kv_cache)
    return output, kv_cache
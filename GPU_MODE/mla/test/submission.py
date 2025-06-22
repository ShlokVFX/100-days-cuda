#!POPCORN leaderboard amd-mla-decode
#!POPCORN gpus MI300

import math
import torch
import torch.nn.functional as F


def precompute_basis(
    seq_len: int, head_dim: int, theta0: float = 10000
) -> tuple[torch.Tensor, torch.Tensor]:
    freq = theta0 ** (
        -torch.arange(0, head_dim, 2, dtype=torch.bfloat16, device="cuda") / head_dim
    )
    pos = torch.arange(seq_len, dtype=torch.bfloat16, device="cuda")
    theta = torch.outer(pos, freq)

    cos = torch.cos(theta).repeat((1, 2))
    sin = torch.sin(theta).repeat((1, 2))

    return cos, sin


def rope(
    x: torch.Tensor, basis: tuple[torch.Tensor, torch.Tensor], pos: int = 0
) -> torch.Tensor:
    cos, sin = basis
    x1, x2 = x.chunk(2, dim=-1)
    return (
        x * cos[pos : pos + x.shape[-2]]
        + torch.cat((-x2, x1), dim=-1) * sin[pos : pos + x.shape[-2]]
    )


#rope = torch.compile(rope, mode="max-autotune")


# JIT-compiled functions for each heavy matmul operation
@torch.jit.script
def q_projection(
    x: torch.Tensor, 
    Q_proj_down_weight: torch.Tensor, 
    Q_proj_up_weight: torch.Tensor
) -> torch.Tensor:
    B, C, d = x.shape
    
    # Ensure contiguous memory layout
    x_flat = x.contiguous().view(-1, d)
    
    # Fuse the two matrix multiplications into one
    # Instead of: x @ Q_proj_down_weight.T @ Q_proj_up_weight.T
    # Pre-compute: Q_proj_up_weight @ Q_proj_down_weight
    combined_weight = Q_proj_up_weight @ Q_proj_down_weight
    
    # Single matrix multiplication instead of two sequential ones
    q_up = (x_flat @ combined_weight.T).view(B, C, 128, 192)  # nh=128, dq_nope+dq_rope=192
    
    return q_up.transpose(1, 2)


@torch.jit.script  
def kv_projection(
    x: torch.Tensor,
    KV_proj_down_weight: torch.Tensor
) -> torch.Tensor:
    B, C, d = x.shape
    return (x.view(-1, d) @ KV_proj_down_weight.T).view(B, C, -1)


@torch.jit.script
def kv_up_projection(
    kv_down_part: torch.Tensor,
    KV_proj_up_weight: torch.Tensor
) -> torch.Tensor:
    B, total_len, dkv = kv_down_part.shape
    kv_up = (kv_down_part.reshape(-1, dkv) @ KV_proj_up_weight.T).view(
        B, total_len, 128, 256  # nh=128, dk_nope+dv=256
    )
    return kv_up.transpose(1, 2)


@torch.jit.script
def attention_matmul(
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    inv_sqrt_dk: float
) -> torch.Tensor:
    return q_flat @ k_flat.transpose(-1, -2) * inv_sqrt_dk


@torch.jit.script
def attention_values(
    attn_weights: torch.Tensor,
    v_flat: torch.Tensor
) -> torch.Tensor:
    return attn_weights @ v_flat


@torch.jit.script
def output_projection(
    y: torch.Tensor,
    wo_weight: torch.Tensor
) -> torch.Tensor:
    B, C, hidden_dim = y.shape
    return (y.view(-1, hidden_dim) @ wo_weight.T).view(B, C, -1)


@torch.no_grad
def compute(
    x: torch.Tensor,
    kv_cache: torch.Tensor,
    prev_len: int,
    Q_proj_down_weight: torch.Tensor,
    Q_proj_up_weight: torch.Tensor,
    KV_proj_down_weight: torch.Tensor,
    KV_proj_up_weight: torch.Tensor,
    wo_weight: torch.Tensor,
):
    # Constants
    nh = 128
    dq_nope = 128
    dq_rope = 64
    dkv = 512
    dk_nope = 128
    dk_rope = 64
    dv = 128

    B, C, d = x.shape
    total_len = prev_len + C
    inv_sqrt_dk = 1.0 / math.sqrt(dk_nope + dq_rope)

    # JIT-compiled Q projection
    q_up = q_projection(x, Q_proj_down_weight, Q_proj_up_weight)

    # Split Q without creating new tensors
    q_nope = q_up[..., :dq_nope]
    q_rope = q_up[..., dq_nope:]

    # Apply RoPE to q_rope
    q_rope = rope(q_rope, basis, prev_len)
    q = torch.cat([q_nope, q_rope], dim=-1)

    # JIT-compiled KV projection
    kv_down = kv_projection(x, KV_proj_down_weight)

    # Update cache in-place
    kv_cache[:, prev_len:total_len] = kv_down

    # Get full context from cache
    kv_full = kv_cache[:, :total_len]
    kv_down_part = kv_full[..., :dkv]
    k_rope_full = kv_full[..., dkv:]

    # JIT-compiled KV up projection
    kv_up = kv_up_projection(kv_down_part, KV_proj_up_weight)

    # Split K and V
    k_nope = kv_up[..., :dk_nope]
    v = kv_up[..., dk_nope:]

    # Apply RoPE to k_rope for full sequence
    k_rope_rotated = rope(k_rope_full, basis, 0)
    k_rope_expanded = k_rope_rotated.unsqueeze(1).expand(-1, nh, -1, -1)
    k = torch.cat([k_nope, k_rope_expanded], dim=-1)

    # Prepare for attention computation
    B_nh = B * nh
    q_flat = q.reshape(B_nh, C, -1)
    k_flat = k.reshape(B_nh, total_len, -1)
    v_flat = v.reshape(B_nh, total_len, dv)

    # JIT-compiled attention computation
    scores = attention_matmul(q_flat, k_flat, inv_sqrt_dk)
    attn_weights = F.softmax(scores, dim=-1)
    y_flat = attention_values(attn_weights, v_flat)

    # Reshape for output
    y = y_flat.reshape(B, nh, C, dv).transpose(1, 2).reshape(B, C, -1)

    # JIT-compiled output projection
    out = output_projection(y, wo_weight)

    return out


basis = precompute_basis(6145, 64)


def custom_kernel(data: tuple) -> tuple:
    config, x, kv_cache = data
    prev_len = kv_cache.seq_len
    out = compute(
        x,
        kv_cache.data,
        prev_len,
        config.Q_proj_down_weight,
        config.Q_proj_up_weight,
        config.KV_proj_down_weight,
        config.KV_proj_up_weight,
        config.wo_weight,
    )
    return out, kv_cache.data
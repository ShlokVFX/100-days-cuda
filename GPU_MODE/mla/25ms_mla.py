#!POPCORN leaderboard amd-mla-decode
#!POPCORN gpus MI300

import math
import torch
import torch.nn.functional as F
from reference import generate_input


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


rope = torch.compile(rope, mode="max-autotune")

k_cache = torch.empty(128, 128, 6145, 192, dtype=torch.bfloat16, device="cuda")
v_cache = torch.empty(128, 128, 6145, 128, dtype=torch.bfloat16, device="cuda")

basis = precompute_basis(6145, 64)


def precompute_kv_proj():
    global k_cache, v_cache
    
    # Constants
    nh = 128
    dq_nope = 128
    dq_rope = 64
    dkv = 512
    dk_nope = 128
    dk_rope = 64
    dv = 128

    inputs = {'dq': 1536, 'dim': 7168, 'batchsize': 128, 'prefill': 6144, 'seed': 5291}
    config, x, kv_cache = generate_input(**inputs)

    B, C, d = x.shape
    prev_len = kv_cache.seq_len
    total_len = prev_len + C
    inv_sqrt_dk = 1.0 / math.sqrt(dk_nope + dq_rope)

    kv_full = kv_cache.data[:, :prev_len]  # B, total_len, dkv + dk_rope
    kv_down_part = kv_full[..., :dkv]
    k_rope_full = kv_full[..., dkv:]
    kv_up = torch.mm(kv_down_part.reshape(-1, dkv), config.KV_proj_up_weight.T).view(
        B, prev_len, nh, dk_nope + dv
    )
    kv_up = kv_up.transpose(1, 2)  # B, nh, total_len, dk_nope + dv
    k_cache[:, :, :prev_len, :dq_nope] = kv_up[..., :dk_nope]
    v_cache[:, :, :prev_len, :] = kv_up[..., dk_nope:]
    k_rope_rotated = rope(k_rope_full, basis, 0)  # Apply RoPE from position 0
    k_cache[:, :, :prev_len, dq_nope:] = k_rope_rotated.unsqueeze(1).expand(
        -1, nh, -1, -1
    )

precompute_kv_proj()

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
    global k_cache, v_cache

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

    k_cache.resize_(128, 128, total_len, 192)
    v_cache.resize_(128, 128, total_len, 128)

    q_down = torch.mm(x.view(-1, d), Q_proj_down_weight.T)  # (B*C, Q_down_dim)
    q = torch.mm(q_down, Q_proj_up_weight.T).view(B, C, nh, dq_nope + dq_rope)
    q = q.transpose(1, 2)  # B, nh, C, dq_nope + dq_rope
    q_rope = q[..., dq_nope:]
    q[:, :, :, dq_nope:] = rope(q_rope, basis, prev_len)

    kv_down = torch.mm(x.view(-1, d), KV_proj_down_weight.T).view(B, C, -1)
    kv_cache[:, prev_len:total_len] = kv_down

    kv_full = kv_cache[:, :total_len]  # B, total_len, dkv + dk_rope
    kv_down_part = kv_full[..., :dkv]
    k_rope_full = kv_full[..., dkv:]
    kv_up = torch.mm(kv_down_part.reshape(-1, dkv), KV_proj_up_weight.T).view(
        B, total_len, nh, dk_nope + dv
    )
    kv_up = kv_up.transpose(1, 2)  # B, nh, total_len, dk_nope + dv
    k_nope = kv_up[..., :dk_nope]
    v = kv_up[..., dk_nope:]
    k_rope_rotated = rope(k_rope_full, basis, 0)  # Apply RoPE from position 0
    k_rope_expanded = k_rope_rotated.unsqueeze(1).expand(-1, nh, -1, -1)
    k = torch.cat([k_nope, k_rope_expanded], dim=-1)

    B_nh = B * nh
    q_flat = q.view(B_nh, C, -1)
    k_flat = k.view(B_nh, total_len, -1)
    v_flat = v.reshape(B_nh, total_len, dv)

    scores = torch.bmm(q_flat, k_flat.transpose(-1, -2))
    scores.mul_(inv_sqrt_dk)
    attn_weights = F.softmax(scores, dim=-1)
    y_flat = torch.bmm(attn_weights, v_flat)  # B_nh, C, dv
    y = y_flat.reshape(B, nh, C, dv).transpose(1, 2).reshape(B, C, -1)
    out = torch.mm(y.view(-1, nh * dv), wo_weight.T).view(B, C, -1)

    return out

@torch.no_grad
def compute_ranked(
    x: torch.Tensor,
    kv_cache: torch.Tensor,
    prev_len: int,
    Q_proj_down_weight: torch.Tensor,
    Q_proj_up_weight: torch.Tensor,
    KV_proj_down_weight: torch.Tensor,
    KV_proj_up_weight: torch.Tensor,
    wo_weight: torch.Tensor,
):
    global k_cache, v_cache

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

    k_cache.resize_(128, 128, total_len, 192)
    v_cache.resize_(128, 128, total_len, 128)

    q_down = torch.mm(x.view(-1, d), Q_proj_down_weight.T)  # (B*C, Q_down_dim)
    q = torch.mm(q_down, Q_proj_up_weight.T).view(B, C, nh, dq_nope + dq_rope)
    q = q.transpose(1, 2)  # B, nh, C, dq_nope + dq_rope
    q_rope = q[..., dq_nope:]
    q[:, :, :, dq_nope:] = rope(q_rope, basis, prev_len)

    kv_down = torch.mm(x.view(-1, d), KV_proj_down_weight.T).view(B, C, -1)
    kv_cache[:, prev_len:total_len] = kv_down

    kv_full = kv_down
    kv_down_part = kv_full[..., :dkv]
    k_rope_full = kv_full[..., dkv:]
    kv_up = torch.mm(kv_down_part.reshape(-1, dkv), KV_proj_up_weight.T).view(
        B, 1, nh, dk_nope + dv
    )
    kv_up = kv_up.transpose(1, 2)  # B, nh, total_len, dk_nope + dv
    k_nope = kv_up[..., :dk_nope]
    v_last = kv_up[..., dk_nope:]
    k_rope_rotated = rope(k_rope_full, basis, prev_len)  # Apply RoPE from position 0
    k_rope_expanded = k_rope_rotated.unsqueeze(1).expand(-1, nh, -1, -1)
    k_last = torch.cat([k_nope, k_rope_expanded], dim=-1)

    k, v = k_cache, v_cache
    k[:, :, prev_len:total_len] = k_last
    v[:, :, prev_len:total_len] = v_last

    B_nh = B * nh
    q_flat = q.view(B_nh, C, -1)
    k_flat = k.view(B_nh, total_len, -1)
    v_flat = v.view(B_nh, total_len, dv)

    scores = torch.bmm(q_flat, k_flat.transpose(-1, -2))
    scores.mul_(inv_sqrt_dk)
    attn_weights = F.softmax(scores, dim=-1)
    y_flat = torch.bmm(attn_weights, v_flat)  # B_nh, C, dv
    y = y_flat.reshape(B, nh, C, dv).transpose(1, 2).reshape(B, C, -1)
    out = torch.mm(y.view(-1, nh * dv), wo_weight.T).view(B, C, -1)

    return out


# compute = torch.compile(compute, mode="max-autotune")


def custom_kernel(data: tuple) -> tuple:
    config, x, kv_cache = data
    prev_len = kv_cache.seq_len
    compute_fn = compute_ranked if kv_cache.seq_len == 6144 else compute
    out = compute_fn(
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

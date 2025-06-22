#!POPCORN leaderboard amd-mixture-of-experts
#!POPCORN gpus MI300

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Dict, Tuple
from task import input_t, output_t


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_warps=8),
    ],
    key=['M', 'N_intermediate', 'K']
)
@triton.jit
def fused_gate_up_kernel(
    x_ptr, w_gate_ptr, w_up_ptr, weight_ptr, intermediate_ptr,
    M, K, N_intermediate,
    stride_xm, stride_xk,
    stride_wg_k, stride_wg_n,
    stride_wu_k, stride_wu_n,
    stride_weight_m,
    stride_im, stride_in,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N_intermediate, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Safe offset calculations
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Early bounds check
    m_mask = offs_am < M
    n_mask = offs_bn < N_intermediate
    
    # FP32 accumulators for stability
    gate_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    up_accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main computation loop
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offs = k * BLOCK_SIZE_K + offs_k
        k_mask = k_offs < K
        
        # Safe pointer calculations
        x_ptrs = x_ptr + offs_am[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_gate_ptrs = w_gate_ptr + k_offs[:, None] * stride_wg_k + offs_bn[None, :] * stride_wg_n
        w_up_ptrs = w_up_ptr + k_offs[:, None] * stride_wu_k + offs_bn[None, :] * stride_wu_n
        
        # Safe loads with proper masking
        x_mask = m_mask[:, None] & k_mask[None, :]
        wgate_mask = k_mask[:, None] & n_mask[None, :]
        wup_mask = k_mask[:, None] & n_mask[None, :]
        
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_gate_vals = tl.load(w_gate_ptrs, mask=wgate_mask, other=0.0)
        w_up_vals = tl.load(w_up_ptrs, mask=wup_mask, other=0.0)
        
        # Accumulate in FP32
        gate_accumulator += tl.dot(x_vals, w_gate_vals)
        up_accumulator += tl.dot(x_vals, w_up_vals)
    
    # Apply SiLU
    gate_silu = gate_accumulator * tl.sigmoid(gate_accumulator)
    
    # Load weights
    weight_ptrs = weight_ptr + offs_am * stride_weight_m
    weight_vals = tl.load(weight_ptrs, mask=m_mask, other=0.0)
    weight_broadcast = weight_vals[:, None]
    
    # Fused result in FP32
    fused_result = gate_silu * up_accumulator * weight_broadcast
    
    # Safe store (FP32 → FP16)
    intermediate_ptrs = intermediate_ptr + offs_am[:, None] * stride_im + offs_bn[None, :] * stride_in
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(intermediate_ptrs, fused_result.to(tl.float16), mask=store_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Early bounds check
    m_mask = offs_am < M
    n_mask = offs_bn < N
    
    # FP32 accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offs = k * BLOCK_SIZE_K + offs_k
        k_mask = k_offs < K
        
        # Safe pointer calculations
        a_ptrs = x_ptr + offs_am[:, None] * stride_xm + k_offs[None, :] * stride_xk
        b_ptrs = w_ptr + k_offs[:, None] * stride_wk + offs_bn[None, :] * stride_wn
        
        # Safe loads
        a_mask = m_mask[:, None] & k_mask[None, :]
        b_mask = k_mask[:, None] & n_mask[None, :]
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        accumulator += tl.dot(a, b)

    # Safe store (FP32 → FP16)
    c_ptrs = out_ptr + offs_am[:, None] * stride_om + offs_bn[None, :] * stride_on
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)


def triton_fused_expert_forward(
    x: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    weight: torch.Tensor
) -> torch.Tensor:
    dtype = torch.float16
    
    # Ensure all tensors are contiguous and FP16
    x = x.contiguous().to(dtype)
    W_gate = W_gate.contiguous().to(dtype)
    W_up = W_up.contiguous().to(dtype)
    W_down = W_down.contiguous().to(dtype)
    weight = weight.contiguous().to(dtype)
    
    M, K = x.shape
    K_gate, N_intermediate = W_gate.shape
    K_up, N_up = W_up.shape
    K_down, N_out = W_down.shape
    
    # Create intermediate buffer in FP16
    intermediate = torch.empty((M, N_intermediate), device=x.device, dtype=dtype)
    
    # Kernel A: Fused gate + up computation
    grid_gate_up = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N_intermediate, META['BLOCK_SIZE_N']),)
    
    fused_gate_up_kernel[grid_gate_up](
        x, W_gate, W_up, weight, intermediate,
        M, K, N_intermediate,
        x.stride(0), x.stride(1),
        W_gate.stride(0), W_gate.stride(1),
        W_up.stride(0), W_up.stride(1),
        weight.stride(0),
        intermediate.stride(0), intermediate.stride(1),
    )
    
    # Kernel B: Standard GEMM intermediate @ W_down
    out = torch.empty((M, N_out), device=x.device, dtype=dtype)
    
    grid_matmul = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N_out, META['BLOCK_SIZE_N']),)
    
    matmul_kernel[grid_matmul](
        intermediate, W_down, out,
        M, N_out, N_intermediate,
        intermediate.stride(0), intermediate.stride(1),
        W_down.stride(0), W_down.stride(1),
        out.stride(0), out.stride(1),
    )
    
    return out


# FP16 I/O with FP32 compute fused_expert_forward
def fused_expert_forward(
    x: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    weight: torch.Tensor
) -> torch.Tensor:
    return triton_fused_expert_forward(x, W_gate, W_up, W_down, weight)


@torch.jit.script
def fused_gating_topk(
    x: torch.Tensor,
    W_g: torch.Tensor,
    top_k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = x @ W_g
    probs = F.softmax(logits, dim=-1)
    topk_scores, topk_indices = torch.topk(probs, k=top_k, dim=-1)
    return topk_scores, topk_indices

@triton.jit
def count_experts_kernel(
    topk_indices_ptr, counts_ptr,
    N: tl.constexpr, top_k: tl.constexpr, num_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_tokens: tl.constexpr = N * top_k
    mask = offset < total_tokens
    
    token_id = offset // top_k
    k_id = offset % top_k
    
    expert_idx = tl.load(topk_indices_ptr + token_id * top_k + k_id, mask=mask, other=num_experts)
    valid_mask = mask & (expert_idx < num_experts)
    
    ONE: tl.constexpr = 1
    tl.atomic_add(counts_ptr + expert_idx, ONE, mask=valid_mask)


@triton.jit
def scatter_sorted_kernel(
    topk_scores_ptr, topk_indices_ptr,
    sorted_token_indices_ptr, sorted_expert_scores_ptr,
    counts_ptr, offsets_ptr,
    N: tl.constexpr, top_k: tl.constexpr, num_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_tokens: tl.constexpr = N * top_k
    mask = offset < total_tokens
    
    token_id = offset // top_k
    k_id = offset % top_k
    
    expert_idx = tl.load(topk_indices_ptr + token_id * top_k + k_id, mask=mask, other=0)
    scores = tl.load(topk_scores_ptr + token_id * top_k + k_id, mask=mask, other=0.0)
    
    valid_mask = mask & (expert_idx < num_experts)
    
    ONE: tl.constexpr = 1
    local_pos = tl.atomic_add(counts_ptr + expert_idx, ONE, mask=valid_mask)
    expert_offset = tl.load(offsets_ptr + expert_idx, mask=valid_mask, other=0)
    final_pos = expert_offset + local_pos
    
    tl.store(sorted_token_indices_ptr + final_pos, token_id, mask=valid_mask)
    tl.store(sorted_expert_scores_ptr + final_pos, scores, mask=valid_mask)


def triton_fused_routing_prep(topk_scores: torch.Tensor, topk_indices: torch.Tensor, 
                             N: int, top_k: int, num_experts: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total_tokens = N * top_k
    
    sorted_token_indices = torch.empty(total_tokens, dtype=torch.long, device=device)
    sorted_expert_scores = torch.empty(total_tokens, dtype=topk_scores.dtype, device=device)
    counts = torch.zeros(num_experts, dtype=torch.long, device=device)
    
    BLOCK_SIZE: int = 256
    grid_size = triton.cdiv(total_tokens, BLOCK_SIZE)
    
    count_experts_kernel[(grid_size,)](
        topk_indices, counts,
        N, top_k, num_experts,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    
    offsets = torch.zeros_like(counts)
    offsets[1:] = counts[:-1].cumsum(0)
    
    temp_counts = torch.zeros_like(counts)
    
    scatter_sorted_kernel[(grid_size,)](
        topk_scores, topk_indices,
        sorted_token_indices, sorted_expert_scores,
        temp_counts, offsets,
        N, top_k, num_experts,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    
    return sorted_token_indices, sorted_expert_scores, counts


def fused_routing_prep(topk_scores: torch.Tensor, topk_indices: torch.Tensor, 
                      N: int, top_k: int, num_experts: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return triton_fused_routing_prep(topk_scores, topk_indices, N, top_k, num_experts, device)


@torch.jit.script
def fused_output_aggregation(
    output: torch.Tensor,
    expert_output: torch.Tensor,
    token_indices: torch.Tensor,
    D: int
) -> torch.Tensor:
    output.scatter_add_(0, token_indices.unsqueeze(1).expand(-1, D), expert_output)
    return output


class MoE(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.num_experts = config["n_routed_experts"]
        self.top_k = config["n_experts_per_token"]
        self.d_hidden = config["d_hidden"]
        
        self.router = None
        self.shared_gate = None
        self.shared_up = None
        self.shared_down = None
        self.gate = [None] * self.num_experts
        self.up = [None] * self.num_experts
        self.down = [None] * self.num_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, T, D = x.shape
            N = B * T
            x_flat = x.view(N, D)

            shared_out = (
                F.silu(x @ self.shared_gate)
                * (x @ self.shared_up)
            ) @ self.shared_down

            topk_scores, topk_indices = fused_gating_topk(x_flat, self.router, self.top_k)
            
            token_indices, expert_scores, counts = fused_routing_prep(
                topk_scores, topk_indices, N, self.top_k, self.num_experts, x.device
            )
            
            cumsum = counts.cumsum(0).cpu().numpy()

            output = torch.zeros_like(x_flat)

            for i in range(self.num_experts):
                start = 0 if i == 0 else cumsum[i - 1]
                end = cumsum[i]
                if start == end:
                    continue

                tokens = token_indices[start:end]
                weights = expert_scores[start:end]
                inputs = x_flat[tokens]

                out = fused_expert_forward(
                    inputs,
                    self.gate[i],
                    self.up[i],
                    self.down[i],
                    weights
                )

                output = fused_output_aggregation(output, out, tokens, D)

            return (shared_out.view(N, D) + output).view(B, T, D)



def custom_kernel(data: input_t) -> output_t:
    x, weights, config = data
    moe = MoE(config)
    moe.router = weights["router.weight"].t()
    moe.shared_gate = weights["shared_experts.0.weight"]
    moe.shared_up = weights["shared_experts.1.weight"]
    moe.shared_down = weights["shared_experts.2.weight"]
    for i in range(config["n_routed_experts"]):
        moe.gate[i] = weights[f"experts.{i}.0.weight"]
        moe.up[i] = weights[f"experts.{i}.1.weight"]
        moe.down[i] = weights[f"experts.{i}.2.weight"]
    return moe(x)
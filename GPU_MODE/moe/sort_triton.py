#!POPCORN leaderboard amd-mixture-of-experts
#!POPCORN gpus MI300

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from task import input_t, output_t


@torch.jit.script
def fused_expert_forward(
    x: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    weight: torch.Tensor
) -> torch.Tensor:
    x = x.contiguous()
    gate = F.silu(x @ W_gate)
    up = x @ W_up
    return (gate * up * weight) @ W_down


@torch.jit.script
def fused_gating_topk(
    x: torch.Tensor,
    W_g: torch.Tensor,
    top_k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused gating network with softmax and top-k selection"""
    logits = x @ W_g
    probs = F.softmax(logits, dim=-1)
    topk_scores, topk_indices = torch.topk(probs, k=top_k, dim=-1)
    return topk_scores, topk_indices

@triton.jit
def count_experts_kernel(
    topk_indices_ptr, counts_ptr,
    N, top_k, num_experts,
    BLOCK_SIZE: tl.constexpr,
):
    """First pass: count occurrences of each expert"""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_tokens = N * top_k
    mask = offset < total_tokens
    
    token_id = offset // top_k
    k_id = offset % top_k
    
    expert_idx = tl.load(topk_indices_ptr + token_id * top_k + k_id, mask=mask, other=num_experts)
    valid_mask = mask & (expert_idx < num_experts)
    
    tl.atomic_add(counts_ptr + expert_idx, 1, mask=valid_mask)


@triton.jit
def scatter_sorted_kernel(
    topk_scores_ptr, topk_indices_ptr,
    sorted_expert_indices_ptr, sorted_token_indices_ptr, sorted_expert_scores_ptr,
    counts_ptr, offsets_ptr,
    N, top_k, num_experts,
    BLOCK_SIZE: tl.constexpr,
):
    """Second pass: scatter to sorted positions"""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_tokens = N * top_k
    mask = offset < total_tokens
    
    token_id = offset // top_k
    k_id = offset % top_k
    
    expert_idx = tl.load(topk_indices_ptr + token_id * top_k + k_id, mask=mask, other=0)
    scores = tl.load(topk_scores_ptr + token_id * top_k + k_id, mask=mask, other=0.0)
    
    valid_mask = mask & (expert_idx < num_experts)
    
    # Get next available position for this expert
    local_pos = tl.atomic_add(counts_ptr + expert_idx, 1, mask=valid_mask)
    expert_offset = tl.load(offsets_ptr + expert_idx, mask=valid_mask, other=0)
    final_pos = expert_offset + local_pos
    
    # Write to sorted positions
    tl.store(sorted_expert_indices_ptr + final_pos, expert_idx, mask=valid_mask)
    tl.store(sorted_token_indices_ptr + final_pos, token_id, mask=valid_mask)
    tl.store(sorted_expert_scores_ptr + final_pos, scores, mask=valid_mask)


def triton_fused_routing_prep(topk_scores: torch.Tensor, topk_indices: torch.Tensor, 
                             N: int, top_k: int, num_experts: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ultra-optimized two-pass implementation"""
    total_tokens = N * top_k
    
    # Output tensors
    sorted_expert_indices = torch.empty(total_tokens, dtype=torch.long, device=device)
    sorted_token_indices = torch.empty(total_tokens, dtype=torch.long, device=device)
    sorted_expert_scores = torch.empty(total_tokens, dtype=topk_scores.dtype, device=device)
    counts = torch.zeros(num_experts, dtype=torch.long, device=device)
    
    # Optimal block size for your hardware
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(total_tokens, BLOCK_SIZE)
    
    # Pass 1: Count experts
    count_experts_kernel[(grid_size,)](
        topk_indices, counts,
        N, top_k, num_experts,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    
    # Compute offsets (CPU is faster for small arrays)
    offsets = torch.zeros_like(counts)
    offsets[1:] = counts[:-1].cumsum(0)
    
    # Reset counts for second pass
    temp_counts = torch.zeros_like(counts)
    
    # Pass 2: Scatter to sorted positions
    scatter_sorted_kernel[(grid_size,)](
        topk_scores, topk_indices,
        sorted_expert_indices, sorted_token_indices, sorted_expert_scores,
        temp_counts, offsets,
        N, top_k, num_experts,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    
    return sorted_expert_indices, sorted_token_indices, sorted_expert_scores, counts


def fused_routing_prep(topk_scores: torch.Tensor, topk_indices: torch.Tensor, 
                      N: int, top_k: int, num_experts: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return triton_fused_routing_prep(topk_scores, topk_indices, N, top_k, num_experts, device)



@torch.jit.script
def fused_output_aggregation(
    output: torch.Tensor,
    expert_output: torch.Tensor,
    token_indices: torch.Tensor,
    D: int
) -> torch.Tensor:
    """Fused output aggregation using scatter_add"""
    output.scatter_add_(0, token_indices.unsqueeze(1).expand(-1, D), expert_output)
    return output


class MoE(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.num_experts = config["n_routed_experts"]
        self.top_k = config["n_experts_per_token"]
        self.d_hidden = config["d_hidden"]
        
        # Attributes that match the custom_kernel interface
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

        # Shared expert computation (keep inline for performance)
        shared_out = (
            F.silu(x @ self.shared_gate)
            * (x @ self.shared_up)
        ) @ self.shared_down

        # Gating network using fused function
        topk_scores, topk_indices = fused_gating_topk(x_flat, self.router, self.top_k)
        
        # Prepare routing using fused function
        expert_indices, token_indices, expert_scores, counts = fused_routing_prep(
            topk_scores, topk_indices, N, self.top_k, self.num_experts, x.device
        )
        
        # Calculate expert boundaries (keep numpy conversion for loop indexing)
        cumsum = counts.cumsum(0).cpu().numpy()

        output = torch.zeros_like(x_flat)

        # Process each expert
        for i in range(self.num_experts):
            start = 0 if i == 0 else cumsum[i - 1]
            end = cumsum[i]
            if start == end:
                continue

            tokens = token_indices[start:end]
            weights = expert_scores[start:end]
            inputs = x_flat[tokens]

            # Apply expert using fused kernel
            out = fused_expert_forward(
                inputs,
                self.gate[i],
                self.up[i],
                self.down[i],
                weights
            )

            # Use fused output aggregation
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
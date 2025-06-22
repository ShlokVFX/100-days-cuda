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


@torch.jit.script
def fused_routing_prep(
    topk_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    N: int,
    top_k: int,
    num_experts: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare routing indices and scores for expert processing"""
    # Flatten and prepare for routing
    expert_indices = topk_indices.reshape(-1)
    expert_scores = topk_scores.reshape(-1, 1)
    token_indices = torch.arange(N, device=device).repeat_interleave(top_k)
    
    # Sort by expert indices for efficient batching
    sorted_idx = expert_indices.argsort()
    expert_indices = expert_indices[sorted_idx]
    token_indices = token_indices[sorted_idx]
    expert_scores = expert_scores[sorted_idx]
    
    # Calculate expert boundaries
    counts = torch.bincount(expert_indices, minlength=num_experts)
    
    return expert_indices, token_indices, expert_scores, counts


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
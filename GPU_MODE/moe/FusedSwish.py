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


class MoE(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.num_experts = config["n_routed_experts"]
        self.top_k = config["n_experts_per_token"]
        self.d_hidden = config["d_hidden"]
        
        # Dictionary-based weight storage for better performance
        self.weights = {
            "W_g": None,
            "W_gate": [None] * self.num_experts,
            "W_up": [None] * self.num_experts,
            "W_down": [None] * self.num_experts,
            "W_shared_gate": None,
            "W_shared_up": None,
            "W_shared_down": None
        }

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        """Load weights into the model"""
        self.weights["W_g"] = weights["router.weight"].t()
        self.weights["W_shared_gate"] = weights["shared_experts.0.weight"]
        self.weights["W_shared_up"] = weights["shared_experts.1.weight"]
        self.weights["W_shared_down"] = weights["shared_experts.2.weight"]
        
        for i in range(self.num_experts):
            self.weights["W_gate"][i] = weights[f"experts.{i}.0.weight"]
            self.weights["W_up"][i] = weights[f"experts.{i}.1.weight"]
            self.weights["W_down"][i] = weights[f"experts.{i}.2.weight"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        N = B * T
        x_flat = x.view(N, D)

        # Shared expert computation
        shared_out = (
            F.silu(x @ self.weights["W_shared_gate"])
            * (x @ self.weights["W_shared_up"])
        ) @ self.weights["W_shared_down"]

        # Gating network
        logits = x_flat @ self.weights["W_g"]
        probs = F.softmax(logits, dim=-1)
        topk_scores, topk_indices = torch.topk(probs, k=self.top_k, dim=-1)
        
        # Prepare for expert routing
        expert_indices = topk_indices.reshape(-1)
        expert_scores = topk_scores.reshape(-1, 1)
        token_indices = torch.arange(N, device=x.device).repeat_interleave(self.top_k)

        # Sort by expert indices for efficient batching
        sorted_idx = expert_indices.argsort()
        expert_indices = expert_indices[sorted_idx]
        token_indices = token_indices[sorted_idx]
        expert_scores = expert_scores[sorted_idx]

        # Calculate expert boundaries
        counts = torch.bincount(expert_indices, minlength=self.num_experts)
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
                self.weights["W_gate"][i],
                self.weights["W_up"][i],
                self.weights["W_down"][i],
                weights
            )

            output.scatter_add_(0, tokens.unsqueeze(1).expand(-1, D), out)

        return (shared_out.view(N, D) + output).view(B, T, D)


def custom_kernel(data: input_t) -> output_t:
    x, weights, config = data
    moe = MoE(config)
    moe.load_weights(weights)
    return moe(x)
#!POPCORN leaderboard amd-mixture-of-experts
#!POPCORN gpus MI300

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from task import input_t, output_t

@torch.jit.script 
def apply_expert(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor, 
    w_down: torch.Tensor,
    weight: torch.Tensor
) -> torch.Tensor:
    x = x.contiguous()
    gated = F.silu(x @ w_gate)
    up = x @ w_up
    return (gated * up * weight) @ w_down

class MoE(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.num_experts = config["n_routed_experts"]
        self.top_k = config["n_experts_per_token"]
        self.weights = {
            "router": None,
            "gate": [None] * self.num_experts,
            "up": [None] * self.num_experts,
            "down": [None] * self.num_experts,
            "shared_gate": None,
            "shared_up": None,
            "shared_down": None
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        N = B * T
        x_flat = x.view(N, D)

        shared = (F.silu(x @ self.weights["shared_gate"]) * (x @ self.weights["shared_up"])) @ self.weights["shared_down"]

        logits = x_flat @ self.weights["router"]
        probs = F.softmax(logits, dim=-1)
        topk_vals, topk_ids = torch.topk(probs, k=self.top_k, dim=-1)

        token_ids = torch.arange(N, device=x.device).repeat_interleave(self.top_k)
        expert_ids = topk_ids.reshape(-1)
        expert_weights = topk_vals.reshape(-1, 1)

        sort_idx = expert_ids.argsort()
        expert_ids = expert_ids[sort_idx]
        token_ids = token_ids[sort_idx]
        expert_weights = expert_weights[sort_idx]

        output = torch.zeros_like(x_flat)
        counts = torch.bincount(expert_ids, minlength=self.num_experts)
        cumsum = counts.cumsum(0).cpu().numpy()

        for i in range(self.num_experts):
            start = 0 if i == 0 else cumsum[i - 1]
            end = cumsum[i]
            if start == end:
                continue

            idx = token_ids[start:end]
            w = expert_weights[start:end]
            inp = x_flat[idx]

            out = apply_expert(
                inp,
                self.weights["gate"][i],
                self.weights["up"][i],
                self.weights["down"][i],
                w
            )

            output.scatter_add_(0, idx.unsqueeze(1).expand(-1, D), out)

        return (shared.view(N, D) + output).view(B, T, D)

    def load(self, w: Dict[str, torch.Tensor]):
        self.weights["router"] = w["router.weight"].t()
        self.weights["shared_gate"] = w["shared_experts.0.weight"]
        self.weights["shared_up"] = w["shared_experts.1.weight"]
        self.weights["shared_down"] = w["shared_experts.2.weight"]
        for i in range(self.num_experts):
            self.weights["gate"][i] = w[f"experts.{i}.0.weight"]
            self.weights["up"][i] = w[f"experts.{i}.1.weight"]
            self.weights["down"][i] = w[f"experts.{i}.2.weight"]

def custom_kernel(data: input_t) -> output_t:
    x, w, cfg = data
    model = MoE(cfg)
    model.load(w)
    return model(x)


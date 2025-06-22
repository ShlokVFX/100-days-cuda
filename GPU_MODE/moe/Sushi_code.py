#!POPCORN leaderboard amd-mixture-of-experts
#!POPCORN gpus MI300

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from task import input_t, output_t


class Expert(nn.Module):
    def __init__(self, config: Dict, d_expert: int = None):
        super().__init__()
        self.config = config
        self.d_hidden = config["d_hidden"]
        self.d_expert = d_expert if d_expert is not None else config["d_expert"]
        self.act_fn = nn.SiLU()

        self.W_gate = None
        self.W_up = None
        self.W_down = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(x @ self.W_gate)
        up = x @ self.W_up
        out = (gate * up) @ self.W_down
        return out


class MoEGate(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.d_hidden = config["d_hidden"]
        self.num_experts = config["n_routed_experts"]
        self.top_k = config["n_experts_per_token"]
        self.W_g = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = x @ self.W_g
        probs = logits.softmax(dim=-1)
        topk_vals, topk_ids = torch.topk(probs, k=self.top_k, dim=-1, sorted=False)
        return topk_ids, topk_vals


class MoE(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.num_experts = config["n_routed_experts"]
        self.top_k = config["n_experts_per_token"]

        self.experts = nn.ModuleList([Expert(config) for _ in range(self.num_experts)])
        self.shared = Expert(config, d_expert=config["d_expert"] * config["n_shared_experts"])
        self.gate = MoEGate(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        N = B * T
        x_flat = x.view(N, D)

        shared_out = self.shared(x)

        expert_ids, expert_weights = self.gate(x)
        expert_ids = expert_ids.view(-1)
        expert_weights = expert_weights.view(-1, 1)
        token_ids = torch.arange(N, device=x.device).repeat_interleave(self.top_k)

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
            out = self.experts[i](inp)
            out.mul_(w)
            output.scatter_add_(0, idx.unsqueeze(1).expand(-1, D), out)

        return (shared_out.view(N, D) + output).view(B, T, D)


def custom_kernel(data: input_t) -> output_t:
    input_tensor, weights, config = data
    model = MoE(config)

    model.gate.W_g = weights["router.weight"].t()
    model.shared.W_gate = weights["shared_experts.0.weight"]
    model.shared.W_up = weights["shared_experts.1.weight"]
    model.shared.W_down = weights["shared_experts.2.weight"]

    for i in range(config["n_routed_experts"]):
        model.experts[i].W_gate = weights[f"experts.{i}.0.weight"]
        model.experts[i].W_up = weights[f"experts.{i}.1.weight"]
        model.experts[i].W_down = weights[f"experts.{i}.2.weight"]

    return model(input_tensor)

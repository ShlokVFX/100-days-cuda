#!POPCORN leaderboard amd-mixture-of-experts
#!POPCORN gpus MI300

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from task import input_t, output_t


class Expert(nn.Module):
    def __init__(self, config: Dict, d_expert: Optional[int] = None):
        super().__init__()
        self.config = config
        self.act_fn = nn.SiLU()
        self.d_hidden: int = config["d_hidden"]
        self.d_expert: int = config["d_expert"] if d_expert is None else d_expert

        self.W_gate = None
        self.W_up = None
        self.W_down = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(x @ self.W_gate)
        out = (gate * (x @ self.W_up)) @ self.W_down
        return out


class MoEGate(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.top_k: int = config["n_experts_per_token"]
        self.num_experts: int = config["n_routed_experts"]
        self.d_hidden: int = config["d_hidden"]

        self.W_g = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = x @ self.W_g
        scores = logits.softmax(dim=-1)
        topk_scores, topk_indices = torch.topk(
            scores, k=self.top_k, dim=-1, sorted=False
        )

        return topk_indices, topk_scores


class MoE(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [Expert(config) for _ in range(config["n_routed_experts"])]
        )
        self.gating_network = MoEGate(config)
        shared_expert_dim = config["d_expert"] * config["n_shared_experts"]
        self.shared_expert = Expert(config=config, d_expert=shared_expert_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_output = self.shared_expert(x)
        expert_indices, expert_scores = self.gating_network(x)
        batch_size, seq_len, hidden_dim = x.shape
        orig_shape = x.shape
        x_flat = x.view(-1, hidden_dim)
        flat_expert_indices = expert_indices.view(-1)
        flat_expert_weights = expert_scores.view(-1, 1)
        routed_output_flat = self.moe_infer(
            x_flat, flat_expert_indices, flat_expert_weights
        )

        routed_output = routed_output_flat.view(*orig_shape)
        return routed_output + shared_output

    @torch.no_grad()
    def moe_infer(
        self,
        x: torch.Tensor,
        flat_expert_indices: torch.Tensor,
        flat_expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        counts = flat_expert_indices.bincount().cpu().numpy()
        tokens_per_expert = counts.cumsum()
        num_per_tok = self.config["n_experts_per_token"]
        token_idxs = idxs // num_per_tok
        for expert_id, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
            if start_idx == end_idx:
                continue

            expert = self.experts[expert_id]
            exp_token_idxs = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idxs]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_reduce_(
                0,
                exp_token_idxs.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce="sum",
            )

        return expert_cache


def custom_kernel(data: input_t) -> output_t:
    input_tensor, weights, config = data
    num_experts = config["n_routed_experts"]
    moe = MoE(config)

    moe.gating_network.W_g = weights["router.weight"].t()

    for i in range(num_experts):
        moe.experts[i].W_gate = weights[f"experts.{i}.0.weight"]
        moe.experts[i].W_up = weights[f"experts.{i}.1.weight"]
        moe.experts[i].W_down = weights[f"experts.{i}.2.weight"]

    moe.shared_expert.W_gate = weights["shared_experts.0.weight"]
    moe.shared_expert.W_up = weights["shared_experts.1.weight"]
    moe.shared_expert.W_down = weights["shared_experts.2.weight"]

    output = moe(input_tensor)

    return output

#!POPCORN leaderboard amd-mixture-of-experts
#!POPCORN gpus MI300

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from task import input_t, output_t
import fused_moe_bf16_asm

class Expert(nn.Module):
    def __init__(self, config: Dict, d_expert: Optional[int] = None):
        super().__init__()
        self.config = config
        self.act_fn = nn.SiLU()
        self.d_hidden: int = config["d_hidden"]
        self.d_expert: int = config["d_expert"] if d_expert is None else d_expert

        self.W_gate = nn.Linear(self.d_hidden, self.d_expert, bias=False)
        self.W_up = nn.Linear(self.d_hidden, self.d_expert, bias=False)
        self.W_down = nn.Linear(self.d_expert, self.d_hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.W_gate(x))
        up = self.W_up(x)
        out = self.W_down(gate * up)
        return out

class MoEGate(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.top_k: int = config["n_experts_per_token"]
        self.num_experts: int = config["n_routed_experts"]
        self.d_hidden: int = config["d_hidden"]
        self.W_g = nn.Linear(self.d_hidden, self.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = F.softmax(self.W_g(x), dim=-1)
        topk_scores, topk_indices = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        return topk_indices, topk_scores

class MoE(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([Expert(config) for _ in range(config["n_routed_experts"])])
        self.gating_network = MoEGate(config)
        shared_expert_dim = config["d_expert"] * config["n_shared_experts"]
        self.shared_expert = Expert(config=config, d_expert=shared_expert_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_output = self.shared_expert(x)
        
        # Fast path using asm_moe if input is in bfloat16 format
        if x.dtype == torch.bfloat16:
            bsz, seq_len, hidden_dim = x.shape
            x_flat = x.reshape(-1, hidden_dim)
            
            # Get expert indices and scores from gating network
            expert_indices, expert_scores = self.gating_network(x)
            expert_indices_flat = expert_indices.reshape(-1, self.config["n_experts_per_token"])
            expert_scores_flat = expert_scores.reshape(-1, self.config["n_experts_per_token"])
            
            # Prepare weights for asm_moe
            num_experts = self.config["n_routed_experts"]
            expert_dim = self.config["d_expert"]
            hidden_dim = self.config["d_hidden"]
            
            # Prepare w1 (gate and up projection weights) - stack them for each expert
            w1 = torch.empty((num_experts, expert_dim * 2, hidden_dim), dtype=torch.bfloat16, device=x.device)
            w2 = torch.empty((num_experts, hidden_dim, expert_dim), dtype=torch.bfloat16, device=x.device)
            
            for i in range(num_experts):
                # Stack gate and up weights together for the g1u1 format
                w1[i, :expert_dim, :] = self.experts[i].W_gate.weight.to(torch.bfloat16)
                w1[i, expert_dim:, :] = self.experts[i].W_up.weight.to(torch.bfloat16)
                w2[i, :, :] = self.experts[i].W_down.weight.t().to(torch.bfloat16)
            
            # Call asm_moe to compute the routed output
            routed_output_flat = fused_moe_bf16_asm.asm_moe(
                hidden_states=x_flat,
                w1=w1,
                w2=w2,
                topk_weight=expert_scores_flat,
                topk_ids=expert_indices_flat
            )
            
            # Reshape to match the input shape
            routed_output = routed_output_flat.reshape(bsz, seq_len, hidden_dim)
            
            return routed_output + shared_output
        else:
            # Original path for non-bfloat16 input
            expert_indices, expert_scores = self.gating_network(x)
            bsz, seq_len, hidden_dim = x.shape
            x_flat = x.view(-1, hidden_dim)
            indices_flat = expert_indices.view(-1)
            scores_flat = expert_scores.view(-1, 1)
            routed_output_flat = self.moe_infer(x_flat, indices_flat, scores_flat)
            routed_output = routed_output_flat.view(*x.shape)
            return routed_output + shared_output

    @torch.no_grad()
    def moe_infer(self, x: torch.Tensor, flat_expert_indices: torch.Tensor, flat_expert_weights: torch.Tensor) -> torch.Tensor:
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
                reduce='sum'
            )
        return expert_cache

def custom_kernel(data: input_t) -> output_t:
    input_tensor, weights, config = data
    num_experts = config["n_routed_experts"]
    moe = MoE(config)

    moe.gating_network.W_g.weight = nn.Parameter(weights['router.weight'])

    for i in range(num_experts):
        moe.experts[i].W_gate.weight = nn.Parameter(weights[f'experts.{i}.0.weight'].t())
        moe.experts[i].W_up.weight = nn.Parameter(weights[f'experts.{i}.1.weight'].t())
        moe.experts[i].W_down.weight = nn.Parameter(weights[f'experts.{i}.2.weight'].t())

    moe.shared_expert.W_gate.weight = nn.Parameter(weights['shared_experts.0.weight'].t())
    moe.shared_expert.W_up.weight = nn.Parameter(weights['shared_experts.1.weight'].t())
    moe.shared_expert.W_down.weight = nn.Parameter(weights['shared_experts.2.weight'].t())

    return moe(input_tensor)
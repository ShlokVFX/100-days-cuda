#!POPCORN leaderboard amd-mixture-of-experts
#!POPCORN gpus MI300

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from task import input_t, output_t

@torch.jit.script
def expert_computation_kernel(expert_input: torch.Tensor, gate_weight: torch.Tensor, 
                              up_weight: torch.Tensor, down_weight: torch.Tensor, 
                              routing_weights: torch.Tensor) -> torch.Tensor:

    gate_proj = expert_input @ gate_weight
    up_proj = expert_input @ up_weight
    intermediate = F.silu(gate_proj) * up_proj
    expert_output = intermediate @ down_weight
    return expert_output * routing_weights


class JITOptimizedMOE(nn.Module):
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        self.num_routed_experts = config["n_routed_experts"]
        self.num_experts_per_token = config["n_experts_per_token"]
        
        self.all_weights = {
            "router": None,
            "experts_gate": [None] * self.num_routed_experts,
            "experts_up": [None] * self.num_routed_experts,
            "experts_down": [None] * self.num_routed_experts,
            "shared_gate": None,
            "shared_up": None,
            "shared_down": None
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        batch_size, seq_len, hidden_dim = x.shape
        total_tokens = batch_size * seq_len
        
        shared_gate_activation = F.silu(x @ self.all_weights["shared_gate"])
        shared_up = x @ self.all_weights["shared_up"]
        router_logits = x @ self.all_weights["router"]
        
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(
            router_probs, k=self.num_experts_per_token, dim=-1, sorted=True
        )
        
        shared_output = (shared_gate_activation * shared_up) @ self.all_weights["shared_down"]
        
        x_flat = x.reshape(total_tokens, hidden_dim)
        flat_topk_indices = topk_indices.reshape(-1)
        flat_topk_probs = topk_probs.reshape(-1, 1)
        
        routed_output = torch.zeros_like(x_flat)
        
        sorted_idx = flat_topk_indices.argsort()
        token_idx = sorted_idx // self.num_experts_per_token
        
        expert_counts = flat_topk_indices.bincount()
        expert_cumsum = expert_counts.cumsum(0).cpu().numpy()
        
        for expert_id in range(self.num_routed_experts):
            start_idx = 0 if expert_id == 0 else expert_cumsum[expert_id - 1]
            end_idx = expert_cumsum[expert_id] if expert_id < len(expert_cumsum) else 0
            
            if start_idx == end_idx:
                continue
                
            current_tokens = token_idx[start_idx:end_idx]
            expert_input = x_flat[current_tokens]
            
            expert_output = expert_computation_kernel(
                expert_input,
                self.all_weights["experts_gate"][expert_id],
                self.all_weights["experts_up"][expert_id],
                self.all_weights["experts_down"][expert_id],
                flat_topk_probs[sorted_idx[start_idx:end_idx]]
            )
            
            expanded_indices = current_tokens.unsqueeze(1).expand(-1, expert_output.size(1))
            routed_output.scatter_add_(0, expanded_indices, expert_output)
            
        return routed_output.reshape(orig_shape) + shared_output

    def load_weights(self, weights):
        self.all_weights["router"] = weights["router.weight"].t()
        
        for i in range(self.num_routed_experts):
            self.all_weights["experts_gate"][i] = weights[f"experts.{i}.0.weight"]
            self.all_weights["experts_up"][i] = weights[f"experts.{i}.1.weight"]
            self.all_weights["experts_down"][i] = weights[f"experts.{i}.2.weight"]
        
        self.all_weights["shared_gate"] = weights["shared_experts.0.weight"]
        self.all_weights["shared_up"] = weights["shared_experts.1.weight"]
        self.all_weights["shared_down"] = weights["shared_experts.2.weight"]


def custom_kernel(data: input_t) -> output_t:
    input_tensor, weights, config = data
    fused_moe_instance = JITOptimizedMOE(config)
    fused_moe_instance.load_weights(weights)    
    output = fused_moe_instance(input_tensor) 
    return output
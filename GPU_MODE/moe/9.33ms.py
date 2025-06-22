#!POPCORN leaderboard amd-mixture-of-experts
#!POPCORN gpus MI300

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from task import input_t, output_t


class TiledMemoryMOE(nn.Module):
    """MOE with shared memory tiling for better cache utilization"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        self.num_routed_experts = config["n_routed_experts"]
        self.num_experts_per_token = config["n_experts_per_token"]
        
        # Optimal tile size for AMD MI300 cache (adjust based on your memory hierarchy)
        self.tile_size = 512  # Process tokens in blocks of 512
        
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
        
        # Shared expert computation
        shared_gate_activation = F.silu(x @ self.all_weights["shared_gate"])
        shared_up = x @ self.all_weights["shared_up"]
        shared_output = (shared_gate_activation * shared_up) @ self.all_weights["shared_down"]
        
        # Router computation
        router_logits = x @ self.all_weights["router"]
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(
            router_probs, k=self.num_experts_per_token, dim=-1, sorted=False
        )
        
        x_flat = x.reshape(total_tokens, hidden_dim)
        flat_topk_indices = topk_indices.reshape(-1)
        flat_topk_probs = topk_probs.reshape(-1, 1)
        
        routed_output = torch.zeros_like(x_flat)
        
        # Process in tiles for better memory locality
        num_tiles = (total_tokens * self.num_experts_per_token + self.tile_size - 1) // self.tile_size
        
        for tile_idx in range(num_tiles):
            start_pos = tile_idx * self.tile_size
            end_pos = min(start_pos + self.tile_size, total_tokens * self.num_experts_per_token)
            
            if start_pos >= end_pos:
                continue
                
            # Get tile data
            tile_expert_indices = flat_topk_indices[start_pos:end_pos]
            tile_expert_probs = flat_topk_probs[start_pos:end_pos]
            tile_token_indices = torch.arange(start_pos, end_pos, device=x.device) // self.num_experts_per_token
            
            # Process experts within this tile
            unique_experts = torch.unique(tile_expert_indices)
            
            for expert_id in unique_experts:
                expert_mask = tile_expert_indices == expert_id
                expert_tokens = tile_token_indices[expert_mask]
                expert_probs = tile_expert_probs[expert_mask]
                
                expert_input = x_flat[expert_tokens]
                
                # Expert computation
                gate_weight = self.all_weights["experts_gate"][expert_id]
                up_weight = self.all_weights["experts_up"][expert_id]
                down_weight = self.all_weights["experts_down"][expert_id]
                
                expert_output = F.silu(expert_input @ gate_weight) * (expert_input @ up_weight) @ down_weight
                expert_output = expert_output * expert_probs
                
                # Accumulate results
                expanded_indices = expert_tokens.unsqueeze(1).expand(-1, expert_output.size(1))
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


class EinsumOptimizedMOE(nn.Module):
    """MOE using einsum for potentially better optimization"""
    
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
        
        # Use einsum for potentially better optimization
        shared_gate_activation = F.silu(torch.einsum('bsh,hd->bsd', x, self.all_weights["shared_gate"]))
        shared_up = torch.einsum('bsh,hd->bsd', x, self.all_weights["shared_up"])
        router_logits = torch.einsum('bsh,he->bse', x, self.all_weights["router"])
        
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(
            router_probs, k=self.num_experts_per_token, dim=-1, sorted=False
        )
        
        shared_output = torch.einsum('bsd,dh->bsh', shared_gate_activation * shared_up, self.all_weights["shared_down"])
        
        x_flat = x.reshape(total_tokens, hidden_dim)
        flat_topk_indices = topk_indices.reshape(-1)
        flat_topk_probs = topk_probs.reshape(-1, 1)
        
        routed_output = torch.zeros_like(x_flat)
        
        # Original expert processing but with einsum where beneficial
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
            
            # Use einsum for expert computation
            gate_weight = self.all_weights["experts_gate"][expert_id]
            up_weight = self.all_weights["experts_up"][expert_id]
            down_weight = self.all_weights["experts_down"][expert_id]
            
            gate_out = torch.einsum('th,hd->td', expert_input, gate_weight)
            up_out = torch.einsum('th,hd->td', expert_input, up_weight)
            expert_output = torch.einsum('td,dh->th', F.silu(gate_out) * up_out, down_weight)
            
            expert_output.mul_(flat_topk_probs[sorted_idx[start_idx:end_idx]])
            
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


@torch.jit.script
def expert_computation_kernel(expert_input: torch.Tensor, gate_weight: torch.Tensor, 
                            up_weight: torch.Tensor, down_weight: torch.Tensor, 
                            routing_weights: torch.Tensor) -> torch.Tensor:
    """JIT compiled expert computation for better optimization"""
    gate_proj = expert_input @ gate_weight
    up_proj = expert_input @ up_weight
    intermediate = F.silu(gate_proj) * up_proj
    expert_output = intermediate @ down_weight
    return expert_output * routing_weights


class JITOptimizedMOE(nn.Module):
    """MOE with JIT compilation for critical paths"""
    
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
            router_probs, k=self.num_experts_per_token, dim=-1, sorted=False
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
            
            # Use JIT compiled kernel
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
    
    # Try JIT optimized version first
    fused_moe_instance = JITOptimizedMOE(config)
    fused_moe_instance.load_weights(weights)
    
    output = fused_moe_instance(input_tensor) 
    return output
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
    """
    Fused expert computation kernel for SwiGLU-style experts.
    Computes: routing_weights * (silu(x @ gate) * (x @ up)) @ down
    """
    gate_proj = expert_input @ gate_weight
    up_proj = expert_input @ up_weight
    intermediate = F.silu(gate_proj) * up_proj
    expert_output = intermediate @ down_weight
    return expert_output * routing_weights

@torch.jit.script
def efficient_token_dispatch(flat_tokens: torch.Tensor, 
                            flat_expert_ids: torch.Tensor,
                            flat_weights: torch.Tensor,
                            num_experts: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Efficiently groups tokens by their assigned experts using sorting.
    This implements the "clever way" mentioned in the hint.
    """
    # Sort by expert ID to group tokens for each expert together
    sorted_indices = flat_expert_ids.argsort()
    sorted_expert_ids = flat_expert_ids[sorted_indices]
    sorted_tokens = flat_tokens[sorted_indices // flat_tokens.size(0)]
    sorted_weights = flat_weights[sorted_indices]
    
    return sorted_tokens, sorted_expert_ids, sorted_weights

class OptimizedMOE(nn.Module):
    """
    Optimized Mixture of Experts with efficient token routing.
    
    Key insight from hint: Different tokens in a sequence route to different experts,
    so we need efficient batching by sorting tokens by their expert assignments.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        self.num_routed_experts = config["n_routed_experts"]
        self.num_experts_per_token = config["n_experts_per_token"]
        
        # Use dictionary-based weight storage like the original
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
        """
        Forward pass implementing efficient sparse expert routing.
        
        The key insight: Use torch.topk to generate sparse "Equation 3":
        output = Σ(i ∈ top_k) gate_weight_i * expert_i(x)
        """
        orig_shape = x.shape
        batch_size, seq_len, hidden_dim = x.shape
        total_tokens = batch_size * seq_len
        
        # Compute shared expert path (always active)
        shared_gate_activation = F.silu(x @ self.all_weights["shared_gate"])
        shared_up = x @ self.all_weights["shared_up"]
        shared_output = (shared_gate_activation * shared_up) @ self.all_weights["shared_down"]
        
        # Reshape for token-level processing
        x_flat = x.view(total_tokens, hidden_dim)
        
        # Compute routing decisions using torch.topk (implements sparse Equation 3)
        router_logits = x_flat @ self.all_weights["router"]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Get top-k experts per token (this creates the sparsity)
        topk_weights, topk_indices = torch.topk(
            router_probs, k=self.num_experts_per_token, dim=-1, sorted=True
        )
        
        # Prepare for efficient expert processing
        flat_topk_indices = topk_indices.view(-1)  # [total_tokens * k]
        flat_topk_weights = topk_weights.view(-1, 1)  # [total_tokens * k, 1]
        
        # Create token indices for scatter operations
        token_indices = torch.arange(total_tokens, device=x.device).unsqueeze(1).expand(-1, self.num_experts_per_token).contiguous().view(-1)
        
        # The "clever way" mentioned in hint: Sort by expert ID for efficient batching
        sorted_indices = flat_topk_indices.argsort()
        sorted_expert_ids = flat_topk_indices[sorted_indices]
        sorted_token_indices = token_indices[sorted_indices]
        sorted_weights = flat_topk_weights[sorted_indices]
        
        # Initialize output
        routed_output = torch.zeros_like(x_flat)
        
        # Process each expert efficiently by grouping tokens
        expert_counts = torch.bincount(sorted_expert_ids, minlength=self.num_routed_experts)
        expert_cumsum = expert_counts.cumsum(0).cpu().numpy()
        
        for expert_id in range(self.num_routed_experts):
            start_idx = 0 if expert_id == 0 else expert_cumsum[expert_id - 1]
            end_idx = expert_cumsum[expert_id] if expert_id < len(expert_cumsum) else 0
            
            if start_idx == end_idx:
                continue  # No tokens routed to this expert
            
            # Get tokens and weights for this expert
            expert_token_indices = sorted_token_indices[start_idx:end_idx]
            expert_weights = sorted_weights[start_idx:end_idx]
            expert_inputs = x_flat[expert_token_indices]
            
            # Compute expert output using fused kernel
            expert_output = expert_computation_kernel(
                expert_inputs,
                self.all_weights["experts_gate"][expert_id],
                self.all_weights["experts_up"][expert_id], 
                self.all_weights["experts_down"][expert_id],
                expert_weights
            )
            
            # Scatter-add to accumulate results
            routed_output.scatter_add_(
                0, 
                expert_token_indices.unsqueeze(1).expand(-1, hidden_dim),
                expert_output
            )
        
        # Combine shared and routed outputs
        total_output = shared_output.view(total_tokens, -1) + routed_output
        return total_output.view(orig_shape)
    
    def load_weights(self, weights: Dict[str, torch.Tensor]):
        """Load pre-trained weights into the model."""
        # Router weights - transpose to match expected format
        self.all_weights["router"] = weights["router.weight"].t()
        
        # Shared expert weights  
        self.all_weights["shared_gate"] = weights["shared_experts.0.weight"]
        self.all_weights["shared_up"] = weights["shared_experts.1.weight"] 
        self.all_weights["shared_down"] = weights["shared_experts.2.weight"]
        
        # Expert weights
        for i in range(self.num_routed_experts):
            self.all_weights["experts_gate"][i] = weights[f"experts.{i}.0.weight"]
            self.all_weights["experts_up"][i] = weights[f"experts.{i}.1.weight"]
            self.all_weights["experts_down"][i] = weights[f"experts.{i}.2.weight"]


def custom_kernel(data: input_t) -> output_t:
    """
    Main entry point implementing the optimized MoE kernel.
    
    The implementation handles the key challenge mentioned in the hint:
    efficiently processing sequences where different tokens route to different experts.
    """
    input_tensor, weights, config = data
    
    # Create and configure MoE instance
    moe_instance = OptimizedMOE(config)
    moe_instance.load_weights(weights)
    
    # Forward pass with efficient token routing
    output = moe_instance(input_tensor)
    
    return output
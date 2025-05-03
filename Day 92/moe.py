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

        self.W_gate = nn.Linear(self.d_hidden, self.d_expert, bias=False)
        self.W_up = nn.Linear(self.d_hidden, self.d_expert, bias=False)
        self.W_down = nn.Linear(self.d_expert, self.d_hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use fused operations where possible
        gate = self.act_fn(self.W_gate(x))
        up = self.W_up(x)
        gate_times_up = gate * up  # Element-wise multiplication
        return self.W_down(gate_times_up)


class MoEGate(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.top_k: int = config["n_experts_per_token"]
        self.num_experts: int = config["n_routed_experts"]
        self.d_hidden: int = config["d_hidden"]

        self.W_g = nn.Linear(self.d_hidden, self.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Calculate router logits
        logits = self.W_g(x)
        
        # Compute softmax scores
        scores = F.softmax(logits, dim=-1)
        
        # Get top-k experts and their scores
        topk_scores, topk_indices = torch.topk(scores, k=self.top_k, dim=-1)
        
        # Normalize the selected expert weights to sum to 1
        topk_scores_sum = topk_scores.sum(dim=-1, keepdim=True)
        topk_scores = topk_scores / topk_scores_sum
        
        # Return original logits for auxiliary loss calculation
        return topk_indices, topk_scores, scores


class MoE(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.top_k = config["n_experts_per_token"]
        self.num_experts = config["n_routed_experts"]
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(config)
            for _ in range(self.num_experts)
        ])
        
        # Create gating network
        self.gating_network = MoEGate(config)
        
        # Create shared expert
        shared_expert_dim = config["d_expert"] * config["n_shared_experts"]
        self.shared_expert = Expert(config=config, d_expert=shared_expert_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process through shared expert
        shared_output = self.shared_expert(x)
        
        # Get routing information
        expert_indices, expert_scores, router_probs = self.gating_network(x)
        
        # Calculate auxiliary losses for training stability
        # 1. Load balancing loss
        router_probs_mean = router_probs.mean(dim=(0, 1))
        aux_loss = self.num_experts * (router_probs_mean * router_probs_mean).sum()
        
        # Get shape information
        batch_size, seq_len, hidden_dim = x.shape
        total_tokens = batch_size * seq_len
        
        # Reshape tensors for efficient processing
        x_flat = x.reshape(total_tokens, hidden_dim)
        expert_indices_flat = expert_indices.reshape(total_tokens, self.top_k)
        expert_scores_flat = expert_scores.reshape(total_tokens, self.top_k)
        
        # Process through experts
        routed_output = self.process_tokens(x_flat, expert_indices_flat, expert_scores_flat)
        routed_output = routed_output.reshape(batch_size, seq_len, hidden_dim)
        
        # Combine outputs from shared expert and routed experts
        final_output = routed_output + shared_output
        
        # Create auxiliary data
        aux_data = {
            "router_probs": router_probs,
            "expert_indices": expert_indices,
            "load_balancing_loss": aux_loss
        }
        
        return final_output, aux_data

    def process_tokens(self, x: torch.Tensor, indices: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Efficiently process tokens through their assigned experts.
        
        Args:
            x: Input tensor of shape [total_tokens, hidden_dim]
            indices: Expert indices of shape [total_tokens, top_k]
            scores: Expert scores of shape [total_tokens, top_k]
            
        Returns:
            Processed tensor of shape [total_tokens, hidden_dim]
        """
        total_tokens, hidden_dim = x.shape
        output = torch.zeros_like(x)
        
        # Process each expert separately to maximize memory efficiency
        for expert_idx in range(self.num_experts):
            # Find all tokens that are routed to this expert
            expert_mask = (indices == expert_idx)
            if not expert_mask.any():
                continue
                
            # Get token indices that need this expert
            token_indices = torch.nonzero(expert_mask.any(dim=1)).squeeze(1)
            if token_indices.numel() == 0:
                continue
                
            # Get the tokens for this expert
            expert_tokens = x[token_indices]
            
            # Get corresponding scores for this expert
            score_mask = expert_mask[token_indices]
            flat_scores = scores[token_indices][score_mask]
            
            # Create a tensor to gather the scores
            expert_scores = torch.zeros(token_indices.size(0), device=x.device, dtype=x.dtype)
            for i, (idx, mask_row) in enumerate(zip(token_indices, score_mask)):
                for j, is_match in enumerate(mask_row):
                    if is_match:
                        expert_scores[i] = scores[idx, j]
                        break
            
            # Process tokens through expert
            expert_output = self.experts[expert_idx](expert_tokens)
            
            # Scale output by expert scores (ensuring they are the same dtype)
            expert_scores = expert_scores.view(-1, 1)
            expert_output = expert_output * expert_scores
            
            # Add to output tensor using indexing instead of index_add_
            for i, idx in enumerate(token_indices):
                output[idx] += expert_output[i]
            
        return output


def custom_kernel(data: input_t) -> output_t:
    """
    Submission template for DeepSeek-style Mixture of Experts using PyTorch.
    
    Args:
        data: Tuple of (input: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
            - input: Input tensor of shape [batch_size, seq_len, hidden_size]
            - weights: Dictionary containing model weights
            - config: Dictionary containing model configuration parameters
            
    Returns:
        Tuple containing:
            - output: Processed tensor [batch_size, seq_len, d_model]
            - aux_data: Dictionary with auxiliary data
    """
    input_tensor, weights, config = data
    
    # Create MoE model
    moe = MoE(config)
    
    # Initialize model weights
    moe.gating_network.W_g.weight = nn.Parameter(weights['router.weight'])
    
    # Initialize expert weights
    for i in range(config["n_routed_experts"]):
        moe.experts[i].W_gate.weight = nn.Parameter(weights[f'experts.{i}.0.weight'].t())
        moe.experts[i].W_up.weight = nn.Parameter(weights[f'experts.{i}.1.weight'].t())
        moe.experts[i].W_down.weight = nn.Parameter(weights[f'experts.{i}.2.weight'].t())
    
    # Initialize shared expert weights
    moe.shared_expert.W_gate.weight = nn.Parameter(weights['shared_experts.0.weight'].t())
    moe.shared_expert.W_up.weight = nn.Parameter(weights['shared_experts.1.weight'].t())
    moe.shared_expert.W_down.weight = nn.Parameter(weights['shared_experts.2.weight'].t())
    
    # Forward pass - don't use mixed precision as it causes type mismatches
    output, aux_data = moe(input_tensor)
    
    return output, aux_data
#!POPCORN leaderboard amd-mixture-of-experts
#!POPCORN gpus MI300

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from task import input_t, output_t

import triton
import triton.language as tl

@triton.jit
def moe_forward_kernel(
    hidden_states_ptr, w1_ptr, w2_ptr, topk_weight_ptr, topk_ids_ptr, output_ptr,
    hidden_dim, expert_dim, num_experts, num_tokens, topk,
    stride_hs, stride_w1, stride_w2, stride_out,
    BLOCK_HIDDEN: tl.constexpr,
):
    # Get program ID for parallelization
    pid = tl.program_id(0)
    # Set offsets for accessing memory
    offs = pid * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
    # Mask for handling boundaries
    mask = offs < hidden_dim
    
    # Initialize output to zeros (unnecessary with the provided implementation)
    # This is now handled by zeroing out the output tensor before kernel call
    
    # Loop through top-k experts for each token
    for k in range(topk):
        # Load expert ID and corresponding weight
        expert_id = tl.load(topk_ids_ptr + pid * topk + k)
        weight = tl.load(topk_weight_ptr + pid * topk + k)
        
        # Load hidden states for current token
        hs = tl.load(hidden_states_ptr + pid * stride_hs + offs, mask=mask, other=0.0)
        
        # Calculate pointers for gate and up projection weights
        w1_gate_ptr = w1_ptr + expert_id * stride_w1 + offs
        w1_up_ptr = w1_gate_ptr + expert_dim
        
        # Compute gate and up projections
        gate = tl.dot(hs, tl.load(w1_gate_ptr, mask=mask, other=0.0))
        up = tl.dot(hs, tl.load(w1_up_ptr, mask=mask, other=0.0))
        
        # Apply SiLU activation and multiply with up projection
        act = tl.math.silu(gate) * up
        
        # Calculate pointer for down projection weights
        w2_ptr_e = w2_ptr + expert_id * stride_w2
        
        # Compute down projection
        out = tl.dot(act, tl.load(w2_ptr_e + offs, mask=mask, other=0.0))
        
        # Scale output by expert weight
        out = out * weight
        
        # Load previous output value and add current expert contribution
        prev = tl.load(output_ptr + pid * stride_out + offs, mask=mask, other=0.0)
        tl.store(output_ptr + pid * stride_out + offs, prev + out, mask=mask)

def triton_moe_forward(hidden_states, w1, w2, topk_weight, topk_ids):
    """
    Accelerated MoE forward pass using Triton.
    
    Args:
        hidden_states: Input tensor [num_tokens, hidden_dim]
        w1: Combined gate and up projection weights [num_experts, expert_dim*2, hidden_dim]
        w2: Down projection weights [num_experts, hidden_dim, expert_dim]
        topk_weight: Expert routing weights [num_tokens, topk]
        topk_ids: Expert indices [num_tokens, topk]
        
    Returns:
        Output tensor [num_tokens, hidden_dim]
    """
    num_tokens, hidden_dim = hidden_states.shape
    expert_dim = w2.shape[-1]
    topk = topk_weight.shape[1]
    
    # Initialize output tensor with zeros
    output = torch.zeros_like(hidden_states)
    
    # Ensure tensors are contiguous for efficient memory access
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    if not w1.is_contiguous():
        w1 = w1.contiguous()
    if not w2.is_contiguous():
        w2 = w2.contiguous()
    
    # Block size for triton kernel
    BLOCK_HIDDEN = min(64, triton.next_power_of_2(hidden_dim))
    
    # Launch kernel
    grid = (num_tokens,)
    moe_forward_kernel[grid](
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        output,
        hidden_dim,
        expert_dim,
        w1.shape[0],  # num_experts
        num_tokens,
        topk,
        hidden_states.stride(0),
        w1.stride(0),
        w2.stride(0),
        output.stride(0),
        BLOCK_HIDDEN=BLOCK_HIDDEN,
    )
    
    return output

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
        self.use_triton = torch.cuda.is_available()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process shared expert (always using standard PyTorch)
        shared_output = self.shared_expert(x)
        
        # Get expert indices and scores from gating network
        expert_indices, expert_scores = self.gating_network(x)
        
        # Reshape input for processing
        bsz, seq_len, hidden_dim = x.shape
        x_flat = x.reshape(-1, hidden_dim)
        
        # Use Triton acceleration when possible (on GPU with bfloat16)
        if self.use_triton and x.device.type == 'cuda' and (x.dtype == torch.bfloat16 or x.dtype == torch.float16):
            # Prepare for Triton kernel
            expert_indices_flat = expert_indices.reshape(-1, self.config["n_experts_per_token"])
            expert_scores_flat = expert_scores.reshape(-1, self.config["n_experts_per_token"])
            
            num_experts = self.config["n_routed_experts"]
            expert_dim = self.config["d_expert"]
            
            # Prepare weight matrices in the format expected by the Triton kernel
            w1 = torch.empty((num_experts, expert_dim * 2, hidden_dim), dtype=x.dtype, device=x.device)
            w2 = torch.empty((num_experts, hidden_dim, expert_dim), dtype=x.dtype, device=x.device)
            
            # Copy weights to combined tensors
            for i in range(num_experts):
                # First half of w1 is gate weights, second half is up projection weights
                w1[i, :expert_dim, :] = self.experts[i].W_gate.weight
                w1[i, expert_dim:, :] = self.experts[i].W_up.weight
                # w2 is down projection weights (transposed for easier matrix multiplication)
                w2[i, :, :] = self.experts[i].W_down.weight.t()
            
            # Run Triton kernel
            routed_output_flat = triton_moe_forward(
                hidden_states=x_flat,
                w1=w1,
                w2=w2,
                topk_weight=expert_scores_flat,
                topk_ids=expert_indices_flat
            )
            
            # Reshape output back to original dimensions
            routed_output = routed_output_flat.reshape(bsz, seq_len, hidden_dim)
        else:
            # Fallback to PyTorch implementation
            indices_flat = expert_indices.view(-1)
            scores_flat = expert_scores.view(-1, 1)
            routed_output_flat = self.moe_infer(x_flat, indices_flat, scores_flat)
            routed_output = routed_output_flat.view(bsz, seq_len, hidden_dim)
        
        # Combine routed and shared outputs
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

    # Load weights
    moe.gating_network.W_g.weight = nn.Parameter(weights['router.weight'])

    for i in range(num_experts):
        moe.experts[i].W_gate.weight = nn.Parameter(weights[f'experts.{i}.0.weight'].t())
        moe.experts[i].W_up.weight = nn.Parameter(weights[f'experts.{i}.1.weight'].t())
        moe.experts[i].W_down.weight = nn.Parameter(weights[f'experts.{i}.2.weight'].t())

    moe.shared_expert.W_gate.weight = nn.Parameter(weights['shared_experts.0.weight'].t())
    moe.shared_expert.W_up.weight = nn.Parameter(weights['shared_experts.1.weight'].t())
    moe.shared_expert.W_down.weight = nn.Parameter(weights['shared_experts.2.weight'].t())

    return moe(input_tensor)
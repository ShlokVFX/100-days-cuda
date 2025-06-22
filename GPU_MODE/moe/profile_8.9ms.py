#!POPCORN leaderboard amd-mixture-of-experts
#!POPCORN gpus MI300

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from task import input_t, output_t
import time

# Lightweight profiling - profile 5th run (after warmup, before final measurements)
PROFILE_COUNTER = 0
TIMING_RESULTS = {}

def maybe_time(name: str, start_time: float, should_profile: bool):
    """Only record timing if profiling is enabled."""
    if should_profile:
        elapsed = (time.perf_counter() - start_time) * 1000
        if name not in TIMING_RESULTS:
            TIMING_RESULTS[name] = []
        TIMING_RESULTS[name].append(elapsed)

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
        global PROFILE_COUNTER
        PROFILE_COUNTER += 1
        should_profile = (PROFILE_COUNTER == 5)  # Profile on 5th run (after warmup)
        
        total_start = time.perf_counter() if should_profile else 0
        
        B, T, D = x.shape
        N = B * T
        x_flat = x.view(N, D)

        # Shared expert - only time if profiling
        shared_start = time.perf_counter() if should_profile else 0
        shared = (F.silu(x @ self.weights["shared_gate"]) * (x @ self.weights["shared_up"])) @ self.weights["shared_down"]
        if should_profile:
            torch.cuda.synchronize()
            maybe_time("shared_expert", shared_start, True)

        # Router computation
        router_start = time.perf_counter() if should_profile else 0
        logits = x_flat @ self.weights["router"]
        probs = F.softmax(logits, dim=-1)
        topk_vals, topk_ids = torch.topk(probs, k=self.top_k, dim=-1)
        if should_profile:
            torch.cuda.synchronize()
            maybe_time("router_total", router_start, True)

        # Token preparation + sorting
        prep_start = time.perf_counter() if should_profile else 0
        token_ids = torch.arange(N, device=x.device).repeat_interleave(self.top_k)
        expert_ids = topk_ids.reshape(-1)
        expert_weights = topk_vals.reshape(-1, 1)
        
        sort_idx = expert_ids.argsort()
        expert_ids = expert_ids[sort_idx]
        token_ids = token_ids[sort_idx]
        expert_weights = expert_weights[sort_idx]
        
        counts = torch.bincount(expert_ids, minlength=self.num_experts)
        cumsum = counts.cumsum(0).cpu().numpy()
        if should_profile:
            torch.cuda.synchronize()
            maybe_time("prep_and_sort", prep_start, True)

        # Expert processing - main bottleneck
        expert_start = time.perf_counter() if should_profile else 0
        output = torch.zeros_like(x_flat)
        
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

        if should_profile:
            torch.cuda.synchronize()
            maybe_time("expert_processing", expert_start, True)

        # Final combination
        result = (shared.view(N, D) + output).view(B, T, D)
        
        if should_profile:
            maybe_time("total_forward", total_start, True)
        
        return result

    def load(self, w: Dict[str, torch.Tensor]):
        self.weights["router"] = w["router.weight"].t()
        self.weights["shared_gate"] = w["shared_experts.0.weight"]
        self.weights["shared_up"] = w["shared_experts.1.weight"]
        self.weights["shared_down"] = w["shared_experts.2.weight"]
        for i in range(self.num_experts):
            self.weights["gate"][i] = w[f"experts.{i}.0.weight"]
            self.weights["up"][i] = w[f"experts.{i}.1.weight"]
            self.weights["down"][i] = w[f"experts.{i}.2.weight"]

def print_timing_summary():
    """Print lightweight timing summary."""
    if not TIMING_RESULTS:
        return
        
    print(f"\n🕒 MoE Bottleneck Analysis (5th run - after warmup):")
    print("-" * 50)
    
    total_time = sum(TIMING_RESULTS.get("total_forward", [0])) / len(TIMING_RESULTS.get("total_forward", [1]))
    
    operations = [
        ("Expert Processing", "expert_processing"),
        ("Shared Expert", "shared_expert"), 
        ("Router Total", "router_total"),
        ("Prep & Sort", "prep_and_sort")
    ]
    
    for name, key in operations:
        if key in TIMING_RESULTS:
            avg_time = sum(TIMING_RESULTS[key]) / len(TIMING_RESULTS[key])
            percentage = (avg_time / total_time * 100) if total_time > 0 else 0
            print(f"{name:<18}: {avg_time:>6.2f}ms ({percentage:>5.1f}%)")
    
    print(f"{'Total':<18}: {total_time:>6.2f}ms (100.0%)")

def custom_kernel(data: input_t) -> output_t:
    x, w, cfg = data
    model = MoE(cfg)
    model.load(w)
    result = model(x)
    
    # Print summary after 5th run (when profiling happened)
    if PROFILE_COUNTER == 5:  # Just finished 5th run
        print_timing_summary()
    
    return result
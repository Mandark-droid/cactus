#!/usr/bin/env python3
"""Inspect Qwen3 MoE tensor shapes to understand the fused expert format."""
from transformers import AutoModelForCausalLM, AutoConfig
import torch

model_id = "kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.1"
print(f"Loading config for {model_id}...")
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
print(f"Model type: {config.model_type}")
print(f"Num layers: {config.num_hidden_layers}")
print(f"Num experts: {config.num_experts}")
print(f"Num experts per tok: {config.num_experts_per_tok}")
print(f"Hidden size: {config.hidden_size}")
print(f"Intermediate size: {config.intermediate_size}")
print(f"MoE intermediate size: {config.moe_intermediate_size}")
print()

print(f"Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
print()

print("=== Expert tensor shapes (layer 0) ===")
for name, param in model.named_parameters():
    if 'layers.0.mlp' in name:
        print(f"  {name}: {param.shape} dtype={param.dtype}")

print()
print("=== All unique tensor name patterns ===")
patterns = set()
for name, param in model.named_parameters():
    # Replace layer numbers with {i} and expert numbers with {e}
    import re
    pattern = re.sub(r'layers\.\d+', 'layers.{i}', name)
    pattern = re.sub(r'experts\.\d+', 'experts.{e}', pattern)
    if pattern not in patterns:
        patterns.add(pattern)
        print(f"  {pattern}: {param.shape}")

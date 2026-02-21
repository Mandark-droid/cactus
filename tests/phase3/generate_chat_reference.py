#!/usr/bin/env python3
"""
Generate HuggingFace reference output for Qwen3 MoE model with chat template.

This script loads the Qwen3 MoE model from HuggingFace, uses the chat template
format, and performs greedy decoding (temperature=0) for 20 tokens.
Prints token IDs at each step with top-5 token IDs and probabilities.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    # Model configuration
    model_name = "kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    print(f"Data type: {dtype}")
    print("-" * 80)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()

    # Qwen3 chat template format
    user_message = "Hello"
    chat_template_prompt = f"<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

    print(f"Chat template prompt:")
    print(f"  {repr(chat_template_prompt)}")
    print()

    # Tokenize the prompt
    input_ids = tokenizer.encode(chat_template_prompt, return_tensors="pt").to(device)
    prompt_length = input_ids.shape[1]

    print(f"Tokenized prompt (length={prompt_length}):")
    print(f"  Token IDs: {input_ids[0].tolist()}")
    print()

    # Print token strings for reference
    print(f"Tokens:")
    for i, token_id in enumerate(input_ids[0].tolist()):
        token_str = tokenizer.decode([token_id])
        print(f"  {i}: {token_id:6d} -> {repr(token_str)}")
    print()

    print("Starting greedy decoding (temperature=0) for 20 tokens:")
    print("-" * 80)

    # Greedy decoding loop
    max_new_tokens = 20
    generated_tokens = []

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids, output_scores=True, return_dict=True)
            logits = outputs.logits[:, -1, :]  # Shape: (1, vocab_size)

        # Get top-5 token probabilities
        probs = F.softmax(logits, dim=-1)
        top5_probs, top5_token_ids = torch.topk(probs[0], k=5)

        # Greedy selection (argmax)
        next_token_id = torch.argmax(logits[0])
        next_token_str = tokenizer.decode([next_token_id])

        generated_tokens.append(next_token_id.item())

        # Print step information
        print(f"Step {step+1:2d}:")
        print(f"  Generated token ID: {next_token_id.item():6d}")
        print(f"  Generated token: {repr(next_token_str)}")
        print(f"  Top-5 candidates:")
        for i, (prob, token_id) in enumerate(zip(top5_probs, top5_token_ids)):
            token_str = tokenizer.decode([token_id])
            print(f"    {i+1}. ID={token_id.item():6d}, prob={prob.item():.6f}, token={repr(token_str)}")
        print()

        # Append generated token to input
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)

    # Decode full sequence
    full_generated_text = tokenizer.decode(input_ids[0].tolist())
    prompt_text = tokenizer.decode(input_ids[0, :prompt_length].tolist())
    generated_text = tokenizer.decode(generated_tokens)

    print("-" * 80)
    print(f"Original prompt:")
    print(f"  {repr(prompt_text)}")
    print()
    print(f"Generated continuation ({len(generated_tokens)} tokens):")
    print(f"  {repr(generated_text)}")
    print()
    print(f"Full generated sequence:")
    print(f"  {repr(full_generated_text)}")
    print()
    print(f"Generated token IDs: {generated_tokens}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate HuggingFace reference outputs for Qwen3 MoE validation.

Usage:
    python generate_reference.py [--model MODEL_ID] [--prompt PROMPT]

Outputs token IDs and greedy-decoded text for comparison with Cactus.
"""
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="Generate HF reference outputs")
    parser.add_argument("--model", default="kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.1",
                        help="HuggingFace model ID")
    parser.add_argument("--prompt", default="Hello",
                        help="Test prompt")
    parser.add_argument("--max-tokens", type=int, default=20,
                        help="Max tokens to generate")
    parser.add_argument("--output", default="reference_output.json",
                        help="Output JSON file")
    args = parser.parse_args()

    print(f"=== HuggingFace Reference Generation ===")
    print(f"Model: {args.model}")
    print(f"Prompt: \"{args.prompt}\"")
    print()

    # Load model and tokenizer
    print("[1/3] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float32
    )
    model.eval()
    print(f"  Model type: {model.config.model_type}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Tokenize
    print("[2/3] Tokenizing...")
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    token_list = input_ids[0].tolist()
    print(f"  Token IDs ({len(token_list)}): {token_list}")
    print()

    # Generate greedily
    print(f"[3/3] Generating {args.max_tokens} tokens (greedy)...")
    generated_ids = []
    all_logits = []

    with torch.no_grad():
        current_ids = input_ids
        past_key_values = None

        for step in range(args.max_tokens):
            outputs = model(
                input_ids=current_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = outputs.logits[:, -1, :]  # [1, vocab_size]
            past_key_values = outputs.past_key_values

            # Greedy: pick argmax
            next_token = logits.argmax(dim=-1).item()
            generated_ids.append(next_token)

            # Store top-5 logits for comparison
            top5_vals, top5_idx = torch.topk(logits[0], 5)
            step_info = {
                "step": step,
                "token_id": next_token,
                "token_text": tokenizer.decode([next_token]),
                "top5_ids": top5_idx.tolist(),
                "top5_logits": [round(v, 4) for v in top5_vals.tolist()],
            }
            all_logits.append(step_info)

            # Print progress
            token_text = tokenizer.decode([next_token])
            print(f"  Step {step:2d}: token={next_token:6d} \"{token_text}\"  "
                  f"top5={top5_idx.tolist()}")

            # Stop on EOS
            if next_token in (tokenizer.eos_token_id, 151643, 151645):
                print(f"  [EOS reached at step {step}]")
                break

            current_ids = torch.tensor([[next_token]])

    print()

    # Print full generation
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"=== Generated Text ===")
    print(f"{args.prompt}{full_text}")
    print()

    print(f"=== Generated Token IDs ===")
    print(generated_ids)
    print()

    # Save reference output
    reference = {
        "model": args.model,
        "prompt": args.prompt,
        "input_token_ids": token_list,
        "generated_token_ids": generated_ids,
        "generated_text": full_text,
        "steps": all_logits,
    }

    with open(args.output, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"Reference saved to {args.output}")

if __name__ == "__main__":
    main()

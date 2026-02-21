"""Re-save Qwen3 MoE router weights in FP16 format (instead of INT8).

The router weights are tiny (16x512 = 8192 params per layer, 12 layers).
INT8 quantization of these small matrices may introduce routing errors.
FP16 preserves exact half-precision values with no quantization loss.

Usage:
    python resave_router_fp16.py <hf_model_id> <cactus_weights_dir>

Example:
    python resave_router_fp16.py kshitijthakkar/loggenix-moe-0.4B-0.2A-sft-s3.1 weights/loggenix-moe-0.4b-0.2a-sft-s3.1
"""

import sys
import struct
import numpy as np
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM
except ImportError:
    print("ERROR: torch and transformers required. Install with: pip install torch transformers")
    sys.exit(1)

CACTUS_MAGIC = b'CACT'
CACTUS_ALIGNMENT = 32

def align_offset(offset, alignment):
    remainder = offset % alignment
    return offset if remainder == 0 else offset + (alignment - remainder)

def save_fp16_weights(data_np, output_path):
    """Save a 2D numpy array in Cactus FP16 format."""
    data = data_np.astype(np.float16)
    shape = list(data.shape)
    ndim = len(shape)
    data_flat = data.flatten()
    data_bytes = data_flat.size * 2  # FP16 = 2 bytes per element

    with open(output_path, 'wb') as f:
        # 84-byte header
        f.write(CACTUS_MAGIC)                          # 4 bytes
        f.write(struct.pack('<I', 0))                   # flags: 0 (no scales, no interleave)
        f.write(struct.pack('<I', CACTUS_ALIGNMENT))    # alignment: 32
        f.write(struct.pack('<I', ndim))                # ndim: 2

        for i in range(4):                              # dims: 4 x 8 bytes
            if i < ndim:
                f.write(struct.pack('<Q', shape[i]))
            else:
                f.write(struct.pack('<Q', 0))

        f.write(struct.pack('<I', 1))                   # precision: 1 = FP16
        f.write(struct.pack('<Q', data_bytes))           # data_bytes
        f.write(struct.pack('<Q', 0))                    # scales_bytes: 0
        f.write(struct.pack('<I', 0))                    # group_size: 0
        f.write(struct.pack('<I', 0))                    # num_groups: 0
        f.write(struct.pack('<Q', shape[0]))             # original_N

        # Padding to alignment
        header_size = 84
        aligned = align_offset(header_size, CACTUS_ALIGNMENT)
        f.write(b'\x00' * (aligned - header_size))

        # Data
        f.write(data_flat.tobytes())

    return data_bytes

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <hf_model_id> <cactus_weights_dir>")
        sys.exit(1)

    model_id = sys.argv[1]
    weights_dir = Path(sys.argv[2])

    if not weights_dir.exists():
        print(f"ERROR: Weights directory not found: {weights_dir}")
        sys.exit(1)

    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    num_layers = model.config.num_hidden_layers

    print(f"Model has {num_layers} layers")
    print(f"Extracting router weights and saving as FP16...\n")

    for i in range(num_layers):
        layer = model.model.layers[i]
        router_weight = layer.mlp.gate.weight.detach().cpu().numpy()

        output_path = weights_dir / f"layer_{i}_moe_router.weights"
        old_size = output_path.stat().st_size if output_path.exists() else 0

        data_bytes = save_fp16_weights(router_weight, output_path)
        new_size = output_path.stat().st_size

        print(f"  Layer {i:2d}: shape={router_weight.shape}, "
              f"old={old_size:,} bytes -> new={new_size:,} bytes, "
              f"range=[{router_weight.min():.4f}, {router_weight.max():.4f}]")

    print(f"\nDone! {num_layers} router weight files saved as FP16.")
    print("Push updated files to Android device and re-run tests.")

if __name__ == '__main__':
    main()

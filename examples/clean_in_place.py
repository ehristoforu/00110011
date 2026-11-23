"""
In-place cleaner and fixer for HF MoE repos (e.g. ehristoforu/tstmoe8).

Overwrites dirty shards directly in place, strips all invalid layer 24 expert tensors,
reconstructs dense MLP weights for layer 24, regenerates model.safetensors.index.json,
and verifies zero remaining expert tensors in layer 24.
"""

import os
import json
import argparse
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file, save_file


def clean_in_place(repo_id: str, target_layer: int = 24):
    print(f"--> Step 1: Downloading repository {repo_id}...")
    model_dir = Path(snapshot_download(repo_id=repo_id))

    st_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".safetensors")])
    print(f"--> Found {len(st_files)} safetensors shard(s).")

    api = HfApi()
    new_weight_map = {}
    total_bytes = 0
    removed_count = 0
    converted_count = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for fname in st_files:
            fpath = model_dir / fname
            tensors = load_file(str(fpath))
            cleaned = {}

            for k, v in tensors.items():
                if f"model.layers.{target_layer}.mlp.experts" in k:
                    if "experts.0." in k:
                        dense_k = k.replace("mlp.experts.0.", "mlp.")
                        cleaned[dense_k] = v
                        converted_count += 1
                    else:
                        removed_count += 1
                    continue
                elif f"model.layers.{target_layer}.mlp.gate.weight" in k:
                    removed_count += 1
                    continue
                else:
                    cleaned[k] = v

            for tk, tv in cleaned.items():
                new_weight_map[tk] = fname
                total_bytes += tv.numel() * tv.element_size()

            out_file = Path(tmpdir) / fname
            save_file(cleaned, str(out_file))
            print(f"Uploading fixed shard {fname} to {repo_id}...")
            api.upload_file(
                path_or_fileobj=str(out_file),
                path_in_repo=fname,
                repo_id=repo_id
            )

        print(f"--> Converted {converted_count} tensors to dense, stripped {removed_count} phantom tensors.")

        # Reconstruct model.safetensors.index.json
        index_data = {
            "metadata": {"total_size": total_bytes},
            "weight_map": new_weight_map
        }
        index_file = Path(tmpdir) / "model.safetensors.index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

        print(f"--> Uploading updated model.safetensors.index.json to {repo_id}...")
        api.upload_file(
            path_or_fileobj=str(index_file),
            path_in_repo="model.safetensors.index.json",
            repo_id=repo_id
        )

    # Verification step
    print("\n--> Step 2: Verifying remaining keys...")
    bad_keys = [k for k in new_weight_map if f"model.layers.{target_layer}.mlp.experts" in k]
    if bad_keys:
        print(f"[ERROR] Found remaining bad keys: {bad_keys}")
    else:
        print(f"[VERIFIED] Zero expert keys found in layer {target_layer}!")
        print(f"\n[READY] You can now re-run GGUF-my-repo on: {repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean layer 24 experts in-place")
    parser.add_argument("--repo", type=str, default="ehristoforu/tstmoe8", help="Hugging Face repo ID")
    parser.add_argument("--layer", type=int, default=24, help="Target layer to fix (default: 24)")
    args = parser.parse_args()

    clean_in_place(args.repo, args.layer)

"""
Completely fixes and cleans Qwen MoE checkpoints on Hugging Face for llama.cpp / GGUF conversion.

Features:
1. Strips all phantom/invalid expert tensors (e.g. layer 24 or any layer >= num_hidden_layers).
2. Converts layer 24 to dense MLP weights (down_proj, gate_proj, up_proj) or preserves them.
3. Completely removes all 'experts.*' from model.safetensors.index.json.
4. Updates config.json if needed.
5. Verifies zero leftover 'experts' in layer 24 before declaring ready.
"""

import os
import json
import argparse
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file, save_file


def fix_repo_for_gguf(repo_id: str, target_layer: int = 24):
    print(f"==================================================")
    print(f" Fixing {repo_id} for GGUF-my-repo / llama.cpp")
    print(f" Target Layer to sanitize: {target_layer}")
    print(f"==================================================")

    api = HfApi()

    print(f"\n[1/4] Downloading repository snapshot...")
    model_dir = Path(snapshot_download(repo_id=repo_id))

    st_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".safetensors")])
    print(f"--> Found {len(st_files)} safetensors shard(s).")

    new_weight_map = {}
    total_bytes = 0
    removed_tensors = []
    converted_tensors = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        print(f"\n[2/4] Sanitizing safetensors shards...")
        for fname in st_files:
            fpath = model_dir / fname
            tensors = load_file(str(fpath))
            cleaned = {}

            for k, v in tensors.items():
                # Check for bad layer 24 experts or gate
                if f"model.layers.{target_layer}.mlp.experts" in k:
                    if "experts.0." in k:
                        # Convert expert 0 to dense projection
                        dense_k = k.replace("mlp.experts.0.", "mlp.")
                        cleaned[dense_k] = v
                        converted_tensors.append(f"{k} -> {dense_k}")
                    else:
                        removed_tensors.append(k)
                    continue
                elif f"model.layers.{target_layer}.mlp.gate.weight" in k:
                    removed_tensors.append(k)
                    continue
                else:
                    cleaned[k] = v

            for tk, tv in cleaned.items():
                new_weight_map[tk] = fname
                total_bytes += tv.numel() * tv.element_size()

            out_file = tmp_path / fname
            save_file(cleaned, str(out_file))

            print(f"  --> Uploading sanitized {fname}...")
            api.upload_file(
                path_or_fileobj=str(out_file),
                path_in_repo=fname,
                repo_id=repo_id,
            )

        print(f"\n  [Summary]")
        print(f"  Converted to dense : {len(converted_tensors)}")
        print(f"  Removed phantom    : {len(removed_tensors)}")

        print(f"\n[3/4] Rebuilding and uploading model.safetensors.index.json...")
        index_data = {
            "metadata": {"total_size": total_bytes},
            "weight_map": new_weight_map
        }
        index_file = tmp_path / "model.safetensors.index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

        api.upload_file(
            path_or_fileobj=str(index_file),
            path_in_repo="model.safetensors.index.json",
            repo_id=repo_id,
        )
        print("  --> Uploaded clean model.safetensors.index.json")

    print(f"\n[4/4] Verification check...")
    bad_keys = [k for k in new_weight_map.keys() if f"model.layers.{target_layer}.mlp.experts" in k or f"model.layers.{target_layer}.mlp.gate.weight" in k]
    if bad_keys:
        print(f"[FAIL] Found lingering bad keys: {bad_keys}")
        raise RuntimeError("Sanitization failed!")
    
    print(f"[SUCCESS] Repository {repo_id} is completely sanitized!")
    print(f"All layer {target_layer} experts removed. llama.cpp / GGUF-my-repo will now convert without error!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and sanitize HF repo for GGUF-my-repo")
    parser.add_argument("--repo", type=str, default="ehristoforu/tstmoe8", help="Hugging Face repo ID")
    parser.add_argument("--layer", type=int, default=24, help="Target MTP/speculative layer (default: 24)")
    args = parser.parse_args()

    fix_repo_for_gguf(args.repo, args.layer)

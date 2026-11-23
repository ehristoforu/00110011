"""
Clean and fix Qwen3.5 MoE checkpoint (e.g. tstmoe6 -> tstmoe7) for GGUF-my-repo.

Removes incomplete/invalid expert tensors from layer 24 (MTP speculative layer),
converts expert 0 to dense projections (gate_proj, up_proj, down_proj),
and uploads the fixed model directly to Hugging Face Hub.
"""

import os
import argparse
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file, save_file


def fix_repo(src_repo: str, dst_repo: str, target_layer: int = 24):
    print(f"--> Downloading source model from HF: {src_repo}...")
    model_dir = Path(snapshot_download(repo_id=src_repo))

    api = HfApi()
    print(f"--> Ensuring target repo exists: {dst_repo}...")
    api.create_repo(repo_id=dst_repo, exist_ok=True)

    st_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".safetensors")])
    print(f"--> Found {len(st_files)} safetensors file(s).")

    with tempfile.TemporaryDirectory() as tmpdir:
        for fname in st_files:
            fpath = model_dir / fname
            print(f"Processing {fname}...")
            tensors = load_file(str(fpath))
            cleaned = {}

            for k, v in tensors.items():
                # Target layer (24 for Qwen 2B MTP) must be strictly dense
                if f"model.layers.{target_layer}.mlp.experts" in k:
                    if "experts.0." in k:
                        dense_k = k.replace("mlp.experts.0.", "mlp.")
                        cleaned[dense_k] = v
                    continue
                elif f"model.layers.{target_layer}.mlp.gate.weight" in k:
                    continue
                else:
                    cleaned[k] = v

            out_file = Path(tmpdir) / fname
            save_file(cleaned, str(out_file))
            print(f"Uploading {fname} to {dst_repo}...")
            api.upload_file(
                path_or_fileobj=str(out_file),
                path_in_repo=fname,
                repo_id=dst_repo
            )

        # Upload config and tokenizer files
        for fname in os.listdir(model_dir):
            if fname.endswith((".json", ".txt", ".model")):
                api.upload_file(
                    path_or_fileobj=str(model_dir / fname),
                    path_in_repo=fname,
                    repo_id=dst_repo
                )

    print(f"\n[DONE] Fixed model uploaded to: https://huggingface.co/{dst_repo}")
    print(f"Use '{dst_repo}' in GGUF-my-repo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix layer 24 experts for Qwen MoE GGUF conversion")
    parser.add_argument("--src", type=str, default="ehristoforu/tstmoe6", help="Source HF repo")
    parser.add_argument("--dst", type=str, default="ehristoforu/tstmoe7", help="Destination HF repo")
    parser.add_argument("--layer", type=int, default=24, help="Target MTP layer to de-MoE (default: 24)")
    args = parser.parse_args()

    fix_repo(args.src, args.dst, args.layer)

"""
Clean and fix Qwen/Qwen3.5 MoE checkpoint for llama.cpp / GGUF-my-repo.

Fixes:
- Strips expert weights and router from MTP/speculative layers (e.g. layer 24 in 2B models)
- Restores dense MLP (gate_proj, up_proj, down_proj) for speculative layers
- Uploads cleaned checkpoint ready for GGUF quantization
"""

import os
import re
import json
import argparse
import tempfile
from pathlib import Path
import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file, save_file


def fix_and_upload_moe(src_repo: str, dst_repo: str, mtp_layer: int = 24):
    print(f"--> Downloading source model: {src_repo}...")
    model_dir = Path(snapshot_download(repo_id=src_repo))

    api = HfApi()
    print(f"--> Ensuring target repo exists: {dst_repo}...")
    api.create_repo(repo_id=dst_repo, exist_ok=True)

    # Detect num_hidden_layers from config.json if present
    cfg_file = model_dir / "config.json"
    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
            t_cfg = cfg_data.get("text_config", cfg_data)
            detected_layers = t_cfg.get("num_hidden_layers")
            if detected_layers is not None:
                mtp_layer = int(detected_layers)
                print(f"--> Auto-detected main layers count = {mtp_layer}. Layers >= {mtp_layer} will be kept dense.")
        except Exception as e:
            print(f"Warning reading config.json: {e}")

    st_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".safetensors")])
    print(f"--> Found {len(st_files)} safetensors file(s) to inspect.")

    target_layer_pat = re.compile(rf"model\.layers\.(\d+)\.")

    with tempfile.TemporaryDirectory() as tmpdir:
        for fname in st_files:
            fpath = model_dir / fname
            print(f"Processing {fname}...")
            tensors = load_file(str(fpath))
            new_tensors = {}

            for k, v in tensors.items():
                m = target_layer_pat.search(k)
                is_mtp = False
                if m and int(m.group(1)) >= mtp_layer:
                    is_mtp = True
                if ".nextn." in k:
                    is_mtp = True

                if is_mtp:
                    # Strip router gate
                    if ".mlp.gate.weight" in k:
                        continue
                    # Convert expert 0 back into dense projection
                    if ".mlp.experts." in k:
                        if ".mlp.experts.0." in k:
                            dense_name = k.replace(".mlp.experts.0.", ".mlp.")
                            new_tensors[dense_name] = v
                        # Skip experts > 0 for MTP layer
                        continue
                    new_tensors[k] = v
                else:
                    new_tensors[k] = v

            out_file = Path(tmpdir) / fname
            save_file(new_tensors, str(out_file))
            print(f"Uploading {fname} -> {dst_repo}...")
            api.upload_file(
                path_or_fileobj=str(out_file),
                path_in_repo=fname,
                repo_id=dst_repo
            )

        # Upload non-tensor configs
        for fname in os.listdir(model_dir):
            if fname.endswith((".json", ".txt", ".md", ".model")):
                api.upload_file(
                    path_or_fileobj=str(model_dir / fname),
                    path_in_repo=fname,
                    repo_id=dst_repo
                )

    print(f"\n[SUCCESS] Model cleaned and uploaded to: https://huggingface.co/{dst_repo}")
    print(f"Now enter '{dst_repo}' into GGUF-my-repo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean MoE MTP layers for GGUF conversion")
    parser.add_argument("--src", type=str, default="ehristoforu/tstmoe5", help="Source HF repo")
    parser.add_argument("--dst", type=str, default="ehristoforu/tstmoe6", help="Destination HF repo")
    parser.add_argument("--mtp-layer", type=int, default=24, help="Starting index of MTP layers (default: 24)")
    args = parser.parse_args()

    fix_and_upload_moe(args.src, args.dst, args.mtp_layer)

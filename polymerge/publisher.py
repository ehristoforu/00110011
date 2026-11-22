import os
from pathlib import Path
from typing import Dict, List, Optional, Any

from polymerge.ui.console import console

class PublisherTool:
    @staticmethod
    def push_to_hub(
        model_dir: str,
        repo_id: str,
        private: bool = False,
        token: Optional[str] = None,
        commit_message: str = "Upload merged model via PolyMerge",
    ):
        """
        Streamlined publication of merged/pruned models to Hugging Face Hub.
        Auto-generates clean metadata and model card.
        """
        from huggingface_hub import HfApi, create_repo
        from polymerge.utils import setup_fast_hf_transfer

        setup_fast_hf_transfer()
        api = HfApi(token=token)
        m_dir = Path(model_dir)

        console.print(f"[bold cyan]Publishing {m_dir} to Hugging Face Hub: {repo_id}...[/bold cyan]")
        create_repo(repo_id=repo_id, private=private, token=token, exist_ok=True)

        card_path = m_dir / "README.md"
        if not card_path.exists():
            card_content = f"""---
license: other
base_model: {repo_id}
tags:
- polymerge
- model-merging
- safetensors
---

# {repo_id.split('/')[-1]}

This model was created using [PolyMerge](https://github.com/ehristoforu/polymerge), the high-performance architecture-agnostic LLM merger and toolkit.
"""
            with open(card_path, "w", encoding="utf-8") as f:
                f.write(card_content)

        api.upload_folder(
            folder_path=str(m_dir),
            repo_id=repo_id,
            commit_message=commit_message,
        )
        console.print(f"[bold green]✓ Successfully uploaded model to: https://huggingface.co/{repo_id}[/bold green]")

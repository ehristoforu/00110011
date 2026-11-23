import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

from polymerge.ui.console import console

class QuantizerTool:
    @staticmethod
    def quantize_gguf(
        model_path: str,
        output_path: str,
        quant_type: str = "Q4_K_M",
        llama_quantize_bin: Optional[str] = None,
    ):
        """
        Automated GGUF post-merge quantization using llama.cpp binary.
        """
        # Look for llama-quantize in PATH or common build dirs
        bin_path = llama_quantize_bin or shutil.which("llama-quantize") or shutil.which("quantize")

        if not bin_path:
            for cand in ["./llama.cpp/build/bin/llama-quantize", "./llama.cpp/llama-quantize"]:
                if os.path.exists(cand):
                    bin_path = cand
                    break

        if not bin_path:
            console.print("[yellow]Notice: 'llama-quantize' binary not found. Saving export instructions.[/yellow]")
            cmd = f"llama-quantize {model_path} {output_path} {quant_type}"
            console.print(f"Run to quantize: [bold cyan]{cmd}[/bold cyan]")
            return False

        console.print(f"[bold cyan]Quantizing {model_path} to {quant_type} via {bin_path}...[/bold cyan]")
        subprocess.run([bin_path, str(model_path), str(output_path), quant_type], check=True)
        console.print(f"[bold green]✓ Quantization complete: {output_path}[/bold green]")
        return True

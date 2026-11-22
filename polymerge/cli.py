import argparse
import sys
from pathlib import Path

def print_custom_help():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]PolyMerge CLI[/bold cyan] - [yellow]High-Performance Universal LLM Model Merger & Optimizer[/yellow]\n"
            "[dim]Architecture-Agnostic | Safetensors & GGUF | Multi-Device | Pure Streaming[/dim]",
            border_style="cyan",
        )
    )

    t = Table(title="CLI Commands & Modes", border_style="magenta", header_style="bold magenta")
    t.add_column("Mode / Command", style="bold green", width=25)
    t.add_column("Arguments & Flags", style="cyan", width=35)
    t.add_column("Description", style="white")

    t.add_row(
        "Standard Merge",
        "<config.yaml> -o <out_dir>",
        "Merge models according to YAML config (12 methods: linear, slerp, nuslerp, karcher, nearswap, ties, della, omnimerge_v2, etc.)"
    )
    t.add_row(
        "MoE Upcycling",
        "<config.yaml> -o <out_dir> --moe",
        "Assemble sparse MoE model from dense experts (Mixtral, Qwen, DeepSeek, Gemma 4 formats)"
    )
    t.add_row(
        "Layer Slicing",
        "<config.yaml> -o <out_dir> --slices",
        "Frankenmerge arbitrary layer slices (e.g. 32L -> 48L stacking)"
    )
    t.add_row(
        "Model Sanity Audit",
        "--audit --base <B> / -o <out_dir>",
        "Audit model for NaN/Inf weights and MoE router matrix condition degeneracy"
    )
    t.add_row(
        "LoRA Extraction",
        "--extract-lora --base <B> --tuned <T> -o <out_dir> [--rank 16] [--alpha 32]",
        "Extract low-rank adapter from weight delta via Truncated SVD"
    )
    t.add_row(
        "LoRA Bake / Merge",
        "--merge-lora --base <B> --adapter <A> -o <out_dir>",
        "Mathematically merge PEFT LoRA adapter directly into base Safetensors"
    )
    t.add_row(
        "LoRA Adapters Merge",
        "--merge-adapters --adapters <A1> <A2> --weights 0.6 0.4 -o <out_dir>",
        "Combine multiple LoRA adapters together (linear or dare_ties)"
    )
    t.add_row(
        "Dense Layer Pruning",
        "--prune-dense --base <B> -o <out_dir> [--drop-fraction 0.2] [--drop-layers ...]",
        "Prune redundant text layers via Cosine Block Similarity & Local Healing"
    )
    t.add_row(
        "MoE Expert Pruning",
        "--prune-moe --base <B> -o <out_dir> --target-experts 64 [--alpha 0.3]",
        "Prune least essential experts with Soft-Transfer Router Recovery (T18 framework)"
    )
    t.add_row(
        "Vocabulary Align",
        "--align-vocab --base <B> --models <M1> <M2> -o <out_dir> [--pad-multiple 64]",
        "Resize and align embedding layers across models with different vocab sizes"
    )
    t.add_row(
        "Vocabulary Truncate",
        "--truncate-vocab <N> --base <B> -o <out_dir>",
        "Truncate bloated vocabulary tensors to exact N tokens"
    )
    t.add_row(
        "Publish to Hub",
        "--push-to-hub <repo_id> -o <out_dir> / --base <B>",
        "Publish merged model folder to Hugging Face Hub with generated model card"
    )
    console.print(t)

    opt_table = Table(title="Global Optimization Flags", border_style="blue", header_style="bold blue")
    opt_table.add_column("Flag", style="bold yellow", width=22)
    opt_table.add_column("Options / Default", style="white", width=20)
    opt_table.add_column("Description", style="dim white")

    opt_table.add_row("--quantize", "Type (e.g. Q4_K_M)", "Auto-quantize output model to GGUF format")
    opt_table.add_row("--device", "auto, cpu, cuda, tpu", "Hardware compute backend (auto-routes to TPU/CUDA/CPU)")
    opt_table.add_row("--chunk-size", "Bytes (def: 2GB)", "Maximum Safetensors shard chunk size in bytes")
    opt_table.add_row("--no-vision-filter", "Flag", "Disable automatic skipping of vision/multimodal encoders")
    opt_table.add_row("-h, --help, help", "Flag / Command", "Display this rich comprehensive guide")
    console.print(opt_table)
    console.print("\n[bold cyan]Documentation & Examples:[/bold cyan] [underline]docs/ARCHITECTURE.md[/underline] | [underline]examples_configs/[/underline]\n")

def main():
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("help", "-h", "--help"):
        print_custom_help()
        sys.exit(0)

    parser = argparse.ArgumentParser(
        prog="polymerge",
        description="PolyMerge: High-performance architecture-agnostic LLM model merger",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="store_true", help="Show this help message and exit")
    parser.add_argument("config", type=str, nargs="?", default=None, help="Path to merge config YAML file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output directory for merged model")
    parser.add_argument("--moe", action="store_true", help="Assemble MoE model from dense experts")
    parser.add_argument("--slices", action="store_true", help="Execute layer slicing / frankenmerging")
    parser.add_argument("--extract-lora", action="store_true", help="Extract LoRA adapter from base and tuned model")
    parser.add_argument("--merge-lora", action="store_true", help="Merge LoRA adapter into base model")
    parser.add_argument("--merge-adapters", action="store_true", help="Merge multiple LoRA adapters together")
    parser.add_argument("--adapters", type=str, nargs="+", default=[], help="Paths to LoRA adapters to merge")
    parser.add_argument("--weights", type=float, nargs="+", default=[], help="Weights for adapters merge")
    parser.add_argument("--method", type=str, default="linear", help="Merge method for adapters (linear, dare_ties)")
    parser.add_argument("--density", type=float, default=1.0, help="Density for DARE adapter merge")
    parser.add_argument("--base", type=str, default=None, help="Base model for LoRA extract/merge")
    parser.add_argument("--tuned", type=str, default=None, help="Tuned model for LoRA extract")
    parser.add_argument("--adapter", type=str, default=None, help="Adapter path for LoRA merge")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank for extraction")
    parser.add_argument("--alpha", type=float, default=32.0, help="LoRA alpha for extraction or MoE router recovery")
    parser.add_argument("--models", type=str, nargs="+", default=[], help="List of source models")
    parser.add_argument("--audit", action="store_true", help="Audit model for NaN/Inf weights and router degeneracy")
    parser.add_argument("--push-to-hub", type=str, default=None, help="Hugging Face repo_id to publish merged model to")
    parser.add_argument("--quantize", type=str, default=None, help="Target GGUF quantization (e.g. Q4_K_M, Q8_0)")
    parser.add_argument("--prune-dense", action="store_true", help="Prune redundant layers from dense model")
    parser.add_argument("--prune-moe", action="store_true", help="Prune least essential experts from MoE model")
    parser.add_argument("--drop-fraction", type=float, default=0.2, help="Fraction of layers to drop for dense pruning")
    parser.add_argument("--drop-layers", type=int, nargs="+", default=None, help="Explicit layer indices to drop")
    parser.add_argument("--target-experts", type=int, default=64, help="Target number of surviving experts for MoE")
    parser.add_argument("--align-vocab", action="store_true", help="Align vocabulary dimensions across models")
    parser.add_argument("--truncate-vocab", type=int, default=None, help="Truncate vocabulary to target size")
    parser.add_argument("--pad-multiple", type=int, default=64, help="Pad vocabulary size to multiple of N")
    parser.add_argument("--device", type=str, default=None, help="Execution device: auto, cpu, cuda, tpu")
    parser.add_argument("--chunk-size", type=int, default=None, help="Shard chunk size in bytes (e.g. 2000000000)")
    parser.add_argument("--no-vision-filter", action="store_true", help="Do not ignore vision / multimodal layers")

    args = parser.parse_args()

    if args.help:
        print_custom_help()
        sys.exit(0)

    if args.audit:
        from polymerge.audit import SanityChecker
        target = args.base or args.output
        if not target:
            sys.exit("Error: --audit requires --base or -o model directory")
        SanityChecker.audit_model(target)
        return

    if args.push_to_hub:
        from polymerge.publisher import PublisherTool
        target = args.base or args.output
        if not target:
            sys.exit("Error: --push-to-hub requires --base or -o model directory")
        PublisherTool.push_to_hub(target, args.push_to_hub)
        return

    if not args.output:
        print_custom_help()
        sys.exit("\n[Error] Missing required parameter: -o / --output")

    if args.prune_dense:
        from polymerge.prune import ModelPruner
        if not args.base:
            sys.exit("Error: --prune-dense requires --base model path")
        ModelPruner.prune_dense_layers(
            args.base,
            args.output,
            drop_layers=args.drop_layers,
            drop_fraction=args.drop_fraction,
            device=args.device or "auto",
            chunk_size_bytes=args.chunk_size or 2_000_000_000,
        )
        return

    if args.prune_moe:
        from polymerge.prune import ModelPruner
        if not args.base:
            sys.exit("Error: --prune-moe requires --base model path")
        ModelPruner.prune_moe_experts(
            args.base,
            args.output,
            target_experts=args.target_experts,
            alpha=args.alpha,
            chunk_size_bytes=args.chunk_size or 2_000_000_000,
        )
        return

    if args.align_vocab:
        from polymerge.tokenizer import TokenizerTool
        if not args.base or not args.models:
            sys.exit("Error: --align-vocab requires --base and --models (list of source models)")
        TokenizerTool.align_vocabularies(args.base, args.models, args.output, pad_to_multiple_of=args.pad_multiple)
        return

    if args.truncate_vocab is not None:
        from polymerge.tokenizer import TokenizerTool
        if not args.base:
            sys.exit("Error: --truncate-vocab requires --base model path")
        TokenizerTool.truncate_vocab(args.base, args.output, target_vocab_size=args.truncate_vocab)
        return

    if args.extract_lora:
        from polymerge.lora import LoRATool
        if not args.base or not args.tuned:
            sys.exit("Error: --extract-lora requires --base and --tuned")
        LoRATool.extract_lora(args.base, args.tuned, args.output, rank=args.rank, alpha=args.alpha)
        return

    if args.merge_lora:
        from polymerge.lora import LoRATool
        if not args.base or not args.adapter:
            sys.exit("Error: --merge-lora requires --base and --adapter")
        LoRATool.merge_lora(args.base, args.adapter, args.output, chunk_size_bytes=args.chunk_size or 2_000_000_000)
        return

    if args.merge_adapters:
        from polymerge.lora import LoRATool
        if not args.adapters or len(args.adapters) < 2:
            sys.exit("Error: --merge-adapters requires at least 2 adapters in --adapters")
        weights = args.weights or [1.0 / len(args.adapters)] * len(args.adapters)
        LoRATool.merge_adapters(args.adapters, weights, args.output, method=args.method, density=args.density)
        return

    if args.slices:
        import yaml
        from polymerge.slicer import LayerSlicer
        with open(args.config, "r", encoding="utf-8") as f:
            slice_data = yaml.safe_load(f)
        slicer = LayerSlicer(slice_data.get("slices", []), chunk_size_bytes=args.chunk_size or 2_000_000_000)
        slicer.execute(args.output)
        return

    if args.moe:
        import yaml
        from polymerge.moe import MoEAssembler
        with open(args.config, "r", encoding="utf-8") as f:
            moe_data = yaml.safe_load(f)
        assembler = MoEAssembler(
            base_model=moe_data.get("base_model"),
            expert_models=moe_data.get("models", []),
            gate_mode=moe_data.get("gate_mode", "uniform_random"),
            target_format=moe_data.get("target_format", "mixtral"),
            shared_expert_model=moe_data.get("shared_expert"),
            pack_experts=moe_data.get("pack_experts", False),
            chunk_size_bytes=args.chunk_size or moe_data.get("chunk_size_bytes", 2_000_000_000),
        )
        assembler.assemble(args.output)
        return

    cfg = MergeConfig.from_yaml(args.config)
    if args.device:
        cfg.device = args.device
    if args.chunk_size:
        cfg.chunk_size_bytes = args.chunk_size
    if args.no_vision_filter:
        cfg.ignore_vision = False

    engine = MergeEngine(cfg)
    engine.run(args.output)

if __name__ == "__main__":
    main()

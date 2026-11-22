import copy
from typing import Dict, List, Optional, Callable, Any
from polymerge.config import MergeConfig
from polymerge.ui.console import console

class EvolutionOptimizer:
    @staticmethod
    def optimize_weights_grid(
        base_config_path: str,
        output_dir: str,
        eval_fn: Callable[[str], float],
        weight_candidates: List[float] = [0.3, 0.5, 0.7],
        density_candidates: List[float] = [0.5, 0.7, 1.0],
    ) -> Dict[str, Any]:
        """
        Lightweight evolutionary/grid search for optimal per-model mixing parameters
        evaluated against a custom scoring function (accuracy, loss, or perplexity).
        """
        base_cfg = MergeConfig.from_yaml(base_config_path)
        best_score = -float("inf")
        best_params = {}

        console.print(f"[bold cyan]Starting Parameter Evolution for {len(base_cfg.models)} models...[/bold cyan]")

        # Search parameter combinations
        for w in weight_candidates:
            for d in density_candidates:
                cfg = copy.deepcopy(base_cfg)
                for m in cfg.models:
                    m.weight = w
                    m.density = d

                candidate_dir = f"{output_dir}_candidate"
                from polymerge.engine import MergeEngine
                engine = MergeEngine(cfg)
                engine.run(candidate_dir)

                score = eval_fn(candidate_dir)
                console.print(f"  • Params (weight={w}, density={d}) -> Score: [bold green]{score:.4f}[/bold green]")

                if score > best_score:
                    best_score = score
                    best_params = {"weight": w, "density": d}

        console.print(f"[bold green]✓ Optimization complete! Best score: {best_score:.4f} with {best_params}[/bold green]")
        return {"best_score": best_score, "best_params": best_params}

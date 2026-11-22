from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel

console = Console()

def print_banner():
    console.print(
        Panel.fit(
            "[bold cyan]PolyMerge[/bold cyan] - [yellow]High-Performance Universal LLM Merger[/yellow]\n"
            "[dim]Architecture-Agnostic | Safetensors & GGUF | Multi-Device | Multi-Method[/dim]",
            border_style="cyan",
        )
    )

def print_plan_table(merge_method: str, base_model: str, models: list, device: str, out_format: str):
    table = Table(title="Merge Execution Plan", border_style="magenta", header_style="bold magenta")
    table.add_column("Property", style="bold yellow")
    table.add_column("Value", style="green")

    table.add_row("Merge Method", merge_method.upper())
    table.add_row("Base Model", str(base_model) if base_model else "[dim]None (Normalized Blending)[/dim]")
    table.add_row("Source Models", f"{len(models)} models")
    table.add_row("Compute Device", str(device).upper())
    table.add_row("Output Format", out_format.upper())
    console.print(table)

    m_table = Table(title="Source Models Breakdown", border_style="blue", header_style="bold blue")
    m_table.add_column("#", justify="right", style="cyan")
    m_table.add_column("Model Path / HF ID", style="white")
    m_table.add_column("Weight", justify="right", style="green")
    m_table.add_column("Density", justify="right", style="yellow")

    for i, m in enumerate(models, start=1):
        m_table.add_row(str(i), m.model, f"{m.weight:.3f}", f"{m.density:.3f}")
    console.print(m_table)

def create_merge_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40, style="cyan", complete_style="bold green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

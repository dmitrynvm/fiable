"""Command-line interface for the model compression tool."""

import sys
from pathlib import Path
from typing import Optional, List
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fiable.config import settings
from fiable.core import download, quantize, evaluate, profile
from fiable.core.metrics import is_baseline_quant, parse_model_identity
from fiable.visual import plots
from fiable import utils


app = typer.Typer(
    name="fiable",
    help="Model compression tool: download, quantize, evaluate, and visualize",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command(name="download")
def download_cmd(
    models: List[str] = typer.Argument(
        None,
        help="Model names to download (default: all)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-download even if exists",
    ),
):
    """
    Download models from Hugging Face.
    
    Example:
        fiable download
        fiable download "Llama 3.1 8B" "Qwen 2.5 7B"
    """
    console.print("[bold cyan]Downloading models...[/bold cyan]\n")
    download.download_models(models if models else None, force=force)


@app.command(name="quantize")
def quantize_cmd(
    models: List[str] = typer.Argument(
        None,
        help="Model names to quantize (default: all downloaded)",
    ),
    quant_types: Optional[str] = typer.Option(
        None,
        "--types",
        "-t",
        help="Comma-separated quantization types",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-quantization even if exists",
    ),
):
    """
    Quantize downloaded models.
    
    Example:
        fiable quantize
        fiable quantize "Llama 3.1 8B" --types "Q4_K_M,Q5_K_M"
    """
    console.print("[bold cyan]Quantizing models...[/bold cyan]\n")
    
    quant_type_list = [q.strip() for q in quant_types.split(",")] if quant_types else None
    quantize.quantize_models(models if models else None, quant_type_list, force=force)


@app.command(name="evaluate")
def evaluate_cmd(
    model_paths: Optional[List[Path]] = typer.Argument(
        None,
        help="GGUF paths (default: all *.gguf in cache/)",
    ),
    perplexity: bool = typer.Option(
        True,
        "--perplexity/--no-perplexity",
        help="Run WikiText-2 perplexity (ctx=512)",
    ),
    long_context: bool = typer.Option(
        True,
        "--long-context/--no-long-context",
        help="Also run perplexity at ctx=2048",
    ),
    benchmarks: bool = typer.Option(
        True,
        "--benchmarks/--no-benchmarks",
        help="Run lm-eval tasks via llama-server",
    ),
    throughput: bool = typer.Option(
        True,
        "--throughput/--no-throughput",
        help="Run throughput benchmark (NVML peak VRAM)",
    ),
    kl: bool = typer.Option(
        True,
        "--kl/--no-kl",
        help="KL divergence vs FP16 (or Q8_0) logits",
    ),
    dataset: str = typer.Option(
        "wikitext",
        "--dataset",
        "-d",
        help="Dataset for perplexity",
    ),
    tasks: Optional[str] = typer.Option(
        None,
        "--tasks",
        "-t",
        help="Comma-separated benchmark tasks (default: mmlu,gsm8k)",
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        help="lm-eval --limit (examples per task); omit for full run",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSON (default: report/evaluation.json)",
    ),
):
    """
    Evaluate quantized models against an FP16 baseline.

    By default, evaluates all GGUF files in cache/ (FP16 + quants).

    Example:
        fiable evaluate
        fiable evaluate cache/*Q4_K_M.gguf --no-benchmarks
        fiable evaluate --tasks gsm8k --limit 50
    """
    console.print("[bold cyan]Evaluating models...[/bold cyan]\n")

    if model_paths is None or len(model_paths) == 0:
        model_paths = evaluate.default_eval_paths()
        if not model_paths:
            console.print("[red]No GGUF models found in cache/[/red]")
            console.print("[yellow]Run 'fiable quantize' first to create quantized models.[/yellow]")
            raise typer.Exit(1)
        console.print(
            f"[dim]Evaluating {len(model_paths)} model(s) in cache/[/dim]\n"
        )

    benchmark_tasks = [t.strip() for t in tasks.split(",")] if tasks else None

    evaluate.evaluate_models(
        model_paths,
        run_perplexity=perplexity,
        run_benchmarks=benchmarks,
        run_throughput=throughput,
        run_long_context=long_context,
        run_kl=kl,
        dataset=dataset,
        benchmark_tasks=benchmark_tasks,
        benchmark_limit=limit,
        output_file=output,
    )


@app.command()
def plot(
    results_file: Optional[Path] = typer.Argument(
        None,
        help="Evaluation JSON (default: report/evaluation.json)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for charts (default: report/)",
    ),
):
    """
    Generate visualization charts from the evaluation report.

    Example:
        fiable plot
        fiable plot report/evaluation.json
    """
    console.print("[bold cyan]Generating charts...[/bold cyan]\n")

    if results_file is None:
        results_file = settings.REPORT_DIR / "evaluation.json"
    if not results_file.exists():
        console.print(f"[red]Results file not found: {results_file}[/red]")
        raise typer.Exit(1)

    plots.generate_all_charts(results_file, output_dir)


@app.command("profile")
def profile_cmd(
    model_paths: Optional[List[Path]] = typer.Argument(
        None,
        help="GGUF paths (default: cache/*fp16.gguf and cache/*Q4_K_M.gguf)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSON file",
    ),
    nsys: bool = typer.Option(
        True,
        "--nsys/--no-nsys",
        help="Try Nsight Systems around a short llama-bench if nsys exists",
    ),
):
    """
    Profile prefill vs decode: TTFT and inter-token latency (FP16 vs Q4_K_M).
    
    Example:
        fiable profile
        fiable profile --no-nsys -o report/exp2_profile.json
    """
    profile.profile_models(model_paths, output_file=output, nsys=nsys)


def _quantized_by_model():
    """Map configured model name -> list of (quant_type, path, size_gb)."""
    grouped: dict = {model.name: [] for model in settings.MODELS}
    unmatched = []
    output_dir = settings.OUTPUT_DIR
    if not output_dir.exists():
        return grouped, unmatched

    for path in sorted(output_dir.glob("*.gguf")):
        _name, quant = parse_model_identity(path)
        if is_baseline_quant(quant):
            continue
        matched = False
        for model in settings.MODELS:
            prefix = f"{model.quant_prefix}-"
            if path.name.startswith(prefix):
                quant_type = path.stem[len(model.quant_prefix) + 1 :]
                size = utils.helpers.get_file_size_gb(path)
                grouped[model.name].append((quant_type, path, size))
                matched = True
                break
        if not matched:
            unmatched.append(path)
    return grouped, unmatched


def _artifact_size_gb(path: Path) -> float:
    """File or directory size in GB."""
    if path.is_file():
        return utils.helpers.get_file_size_gb(path)
    if not path.is_dir():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 ** 3)


_SOURCE_TYPE_SUFFIXES = ("BF16", "FP32", "FP16", "F32", "F16")


def _source_type_label(path: Path) -> str:
    """Precision label from a source GGUF/dir name (FP16, BF16, ...)."""
    stem = path.stem.upper()
    for token in _SOURCE_TYPE_SUFFIXES:
        if stem.endswith(f"-{token}") or stem.endswith(f"_{token}"):
            return token
    return "FP16"


@app.command("cache")
def cache_cmd():
    """
    List cached artifacts: Model, Status (type), Size, File.
    
    Example:
        fiable cache
    """
    quantized, unmatched = _quantized_by_model()

    table = Table()
    table.add_column("Model", style="cyan")
    table.add_column("Status")
    table.add_column("Size (GB)", justify="right")
    table.add_column("File", style="dim", no_wrap=True)

    for model in settings.MODELS:
        model_dir = settings.CACHE_DIR / model.local_dir
        fp16_path = settings.CACHE_DIR / model.fp16_filename
        if fp16_path.exists():
            source_path = fp16_path
        elif model_dir.exists():
            source_path = model_dir
        else:
            source_path = None

        if source_path is not None:
            size = _artifact_size_gb(source_path)
            table.add_row(
                model.name,
                f"[green]{_source_type_label(source_path)}[/green]",
                f"{size:.2f}",
                source_path.name,
            )
        else:
            table.add_row(
                model.name,
                "[dim]None[/dim]",
                "—",
                "—",
            )

        files_by_type = {q: (path, size) for q, path, size in quantized.get(model.name, [])}
        for quant_type in settings.QUANT_TYPES:
            if quant_type not in files_by_type:
                continue
            path, size = files_by_type[quant_type]
            table.add_row(
                model.name,
                f"[magenta]{quant_type}[/magenta]",
                f"{size:.2f}",
                path.name,
            )

    for path in unmatched:
        size = utils.helpers.get_file_size_gb(path)
        stem = path.stem
        quant_type = next(
            (q for q in settings.QUANT_TYPES if stem.endswith(f"-{q}")),
            "unknown",
        )
        table.add_row("Other", f"[magenta]{quant_type}[/magenta]", f"{size:.2f}", path.name)

    datasets_dir = settings.DATASETS_DIR
    if datasets_dir.exists():
        for path in sorted(datasets_dir.iterdir()):
            if not path.is_file():
                continue
            size = utils.helpers.get_file_size_gb(path)
            table.add_row(
                "dataset",
                "[cyan]cached[/cyan]",
                f"{size:.2f}",
                path.name,
            )

    console.print(table)


@app.callback()
def main():
    """
    Fiable Model Compression Tool CLI
    
    A comprehensive tool for downloading, quantizing, evaluating, and visualizing
    large language model compression.
    """
    pass


def run():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    run()

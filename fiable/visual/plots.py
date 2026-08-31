"""Visualization functionality for compression analysis."""

import json
from pathlib import Path
from typing import List, Dict, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rich.console import Console

from fiable.utils import helpers
from fiable.config.settings import STORE_DIR, CHARTS_DIR, CHART_DPI, CHART_COLORS, MODELS, format_precision
from fiable.core.evaluate import EvaluationResult
from fiable.core.metrics import annotate_relative_metrics, read_gguf_parameter_count


console = Console()

sns.set_theme(style="whitegrid", palette=CHART_COLORS, font_scale=1.05)
plt.rcParams["figure.dpi"] = CHART_DPI
plt.rcParams["savefig.dpi"] = CHART_DPI
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.edgecolor"] = "#888888"
plt.rcParams["grid.color"] = "#D0D0D0"
plt.rcParams["grid.linestyle"] = "-"

_BAR_EDGE = "#333333"
_ANNOTATE = "#777777"
_SERIES = (
    {"color": "#1f77b4", "ls": "-", "marker": "o"},
    {"color": "#d62728", "ls": "-", "marker": "o"},
    {"color": "#2ca02c", "ls": "-", "marker": "o"},
    {"color": "#ff7f0e", "ls": "-", "marker": "o"},
    {"color": "#9467bd", "ls": "-", "marker": "o"},
    {"color": "#8c564b", "ls": "-", "marker": "o"},
)


def _palette(n: int):
    n = max(int(n), 1)
    colors = [s["color"] for s in _SERIES]
    if n > len(colors):
        colors.extend(CHART_COLORS[len(colors):])
        colors.extend(sns.color_palette("tab10", n_colors=n).as_hex())
    return colors[:n]


def _series_for(models) -> Dict[str, dict]:
    """Stable color per model: Llama = blue, Qwen = red."""
    names = list(dict.fromkeys(models))
    ordered = [n for n in names if "llama" in n.lower()]
    ordered += [n for n in names if "qwen" in n.lower() and n not in ordered]
    ordered += [n for n in names if n not in ordered]
    return {name: _SERIES[i % len(_SERIES)] for i, name in enumerate(ordered)}


def _model_colors(models) -> Dict[str, str]:
    return {name: style["color"] for name, style in _series_for(models).items()}


def _model_linestyles(models) -> Dict[str, str]:
    return {name: style["ls"] for name, style in _series_for(models).items()}


def _model_markers(models) -> Dict[str, str]:
    return {name: style["marker"] for name, style in _series_for(models).items()}


def _pretty_model(name: str) -> str:
    """llama-3.1-8b-instruct → Llama 3.1 8B."""
    if not name:
        return name
    key = name.lower()
    for cfg in MODELS:
        prefix = str(getattr(cfg, "quant_prefix", "") or "").lower()
        if prefix and prefix in key:
            return cfg.name
        if cfg.name.lower() == key:
            return cfg.name
    label = name
    for suffix in ("-instruct", "_instruct", "-chat", "_chat"):
        if label.lower().endswith(suffix):
            label = label[: -len(suffix)]
    label = label.replace("-", " ").replace("_", " ").strip()
    parts = []
    for token in label.split():
        low = token.lower()
        if low.startswith("llama"):
            parts.append("Llama")
        elif low.startswith("qwen"):
            parts.append("Qwen")
        else:
            parts.append(token.upper() if any(c.isdigit() for c in token) else token.title())
    return " ".join(parts) or name


def _pretty_quant(quant: str) -> str:
    """Q4_K_M → W4A16."""
    return format_precision(quant) if quant else quant


_PRECISION_ORDER = ("W32A32", "W16A16", "W8A16", "W6A16", "W5A16", "W4A16", "W3A16", "W2A16")
_TASK_ORDER = ("mmlu", "gsm8k", "humaneval")
_TASK_MARKERS = ("o", "s", "^", "D", "v", "P")
_TASK_LINESTYLES = ("-", "--", ":", "-.", "-", "--")


def _precision_rank(label: str) -> tuple:
    if label in _PRECISION_ORDER:
        return (_PRECISION_ORDER.index(label), label)
    return (len(_PRECISION_ORDER), str(label or ""))


def _task_rank(task: str) -> tuple:
    if task in _TASK_ORDER:
        return (_TASK_ORDER.index(task), task)
    return (len(_TASK_ORDER), str(task or ""))


def _task_style(task: str, tasks: List[str]) -> dict:
    ordered = sorted(dict.fromkeys(tasks), key=_task_rank)
    idx = ordered.index(task) if task in ordered else 0
    return {
        "marker": _TASK_MARKERS[idx % len(_TASK_MARKERS)],
        "ls": _TASK_LINESTYLES[idx % len(_TASK_LINESTYLES)],
    }


def _finite(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _accuracy_frame(
    results: List[EvaluationResult],
    *,
    require_delta: bool = False,
    require_throughput: bool = False,
) -> Optional[pd.DataFrame]:
    rows = []
    for r in results:
        for task, score in (r.benchmarks or {}).items():
            acc = _finite(score)
            if acc is None:
                continue
            delta = _finite((r.delta_acc or {}).get(task))
            if require_delta and delta is None:
                continue
            tok_s = _finite(r.throughput_tokens_per_sec)
            if require_throughput and tok_s is None:
                continue
            rows.append({
                "model": _pretty_model(r.model_name),
                "quant": _pretty_quant(r.quantization_type),
                "task": task,
                "acc": acc,
                "delta_acc": delta,
                "size_gb": r.file_size_gb,
                "tok_s": tok_s,
            })
    if not rows:
        return None
    return pd.DataFrame(rows)


def _draw_acc_series(ax, df: pd.DataFrame, xcol: str, ycol: str, annotate: bool = True) -> None:
    models = list(dict.fromkeys(df["model"]))
    tasks = sorted(dict.fromkeys(df["task"]), key=_task_rank)
    styles = _series_for(models)
    for model in models:
        st = styles[model]
        for task in tasks:
            part = df[(df["model"] == model) & (df["task"] == task)].copy()
            if part.empty:
                continue
            ts = _task_style(task, tasks)
            part = part.sort_values(xcol)
            ax.plot(
                part[xcol],
                part[ycol],
                marker=ts["marker"],
                markersize=8,
                markeredgecolor=st["color"],
                markerfacecolor=st["color"],
                markeredgewidth=1.2,
                label=f"{model} {task}",
                color=st["color"],
                linestyle=ts["ls"],
                linewidth=1.8,
            )
            if annotate:
                for _, row in part.iterrows():
                    ax.annotate(
                        row["quant"],
                        (row[xcol], row[ycol]),
                        fontsize=8,
                        xytext=(4, 6),
                        textcoords="offset points",
                        color=_ANNOTATE,
                    )


def _style_axes(ax, y_only: bool = False) -> None:
    ax.set_facecolor("white")
    ax.grid(True, axis="y" if y_only else "both", linestyle="-", color="#D0D0D0", alpha=0.85)
    sns.despine(ax=ax, offset=0, trim=False)


def _resolve_model_path(path_str: str) -> str:
    """Map legacy cache/ and /workspace/output/... paths onto store/ if needed."""
    path = Path(path_str)
    if path.exists():
        return str(path)
    name = path.name
    for candidate in (
        STORE_DIR / name,
        Path.cwd() / "store" / name,
        Path.cwd() / "cache" / name,
        Path("/workspace/cache") / name,
    ):
        if candidate.exists():
            return str(candidate)
    return path_str


def load_results(results_file: Path) -> List[EvaluationResult]:
    """Load evaluation results from JSON file."""
    import dataclasses
    data = helpers.load_json(results_file)
    if isinstance(data, dict):
        data = data.get("results", data.get("summary", []))
    aliases = {
        "speed_tokens_per_sec": "throughput_tokens_per_sec",
        "speed_latency_ms": "throughput_latency_ms",
        "speed_memory_gb": "throughput_memory_gb",
        "speed_error": "throughput_error",
    }
    allowed = {f.name for f in dataclasses.fields(EvaluationResult)}
    results = []
    for item in data:
        item = dict(item)
        if "model_path" in item and item["model_path"]:
            item["model_path"] = _resolve_model_path(str(item["model_path"]))
        for old, new in aliases.items():
            if new not in item and old in item:
                item[new] = item[old]
            item.pop(old, None)
        item = {k: v for k, v in item.items() if k in allowed}
        results.append(EvaluationResult(**item))
    for result in results:
        if not result.n_params:
            path = Path(result.model_path)
            if path.exists():
                result.n_params = read_gguf_parameter_count(path)
    annotate_relative_metrics(results)
    return results


def plot_size_vs_quality(
    results: List[EvaluationResult],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Plot size vs quality (perplexity) trade-off curve.
    
    Args:
        results: List of evaluation results
        output_path: Optional output path for the plot
        
    Returns:
        Path to saved plot
    """
    console.print("[cyan]Creating perplexity vs size plot...[/cyan]")
    
    # Filter results with perplexity data
    valid_results = [r for r in results if r.perplexity is not None]
    
    if not valid_results:
        console.print("[yellow]No perplexity data available for plotting[/yellow]")
        return None
    
    # Create DataFrame
    df = pd.DataFrame([
        {
            "model": _pretty_model(r.model_name),
            "quant": _pretty_quant(r.quantization_type),
            "size_gb": r.file_size_gb,
            "perplexity": r.perplexity,
        }
        for r in valid_results
    ])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot points for each model
    unique_models = list(df["model"].unique())
    styles = _series_for(unique_models)

    for model in unique_models:
        model_df = df[df["model"] == model]
        st = styles[model]
        ax.plot(
            model_df["size_gb"],
            model_df["perplexity"],
            marker=st["marker"],
            markersize=8,
            markeredgecolor=st["color"],
            markerfacecolor=st["color"],
            markeredgewidth=1.2,
            label=model,
            color=st["color"],
            linestyle=st["ls"],
            linewidth=1.8,
        )

        for _, row in model_df.iterrows():
            ax.annotate(
                row["quant"],
                (row["size_gb"], row["perplexity"]),
                xytext=(5, 6),
                textcoords="offset points",
                fontsize=8,
                color=_ANNOTATE,
            )

    ax.set_xlabel("Size (GB)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Perplexity", fontsize=12, fontweight="bold")
    ax.set_title("Perplexity vs Size", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10, frameon=True)
    _style_axes(ax)
    ax.invert_yaxis()
    return _finish_plot(df, output_path, "perplexity_vs_size.png")


def plot_throughput_comparison(
    results: List[EvaluationResult],
    output_path: Optional[Path] = None,
) -> Path:
    """Bar chart of decode throughput by model and quantization."""
    console.print("[cyan]Creating throughput comparison plot...[/cyan]")

    rows = []
    for r in results:
        if r.throughput_tokens_per_sec is None:
            continue
        rows.append({
            "model": _pretty_model(r.model_name),
            "quant": _pretty_quant(r.quantization_type),
            "decode": r.throughput_tokens_per_sec,
        })
    if not rows:
        console.print("[yellow]No throughput data available for plotting[/yellow]")
        return None

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 7))
    unique_models = list(df["model"].unique())
    colors = _model_colors(unique_models)
    quants = list(dict.fromkeys(df["quant"]))
    width = 0.8 / max(len(unique_models), 1)
    x = range(len(quants))
    for i, model in enumerate(unique_models):
        model_df = df[df["model"] == model].set_index("quant")
        heights = [float(model_df.loc[q, "decode"]) if q in model_df.index else 0 for q in quants]
        offset = width * (i - (len(unique_models) - 1) / 2)
        ax.bar(
            [j + offset for j in x],
            heights,
            width,
            label=model,
            color=colors[model],
            edgecolor=_BAR_EDGE,
            linewidth=0.7,
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(quants, rotation=30, ha="right")
    ax.set_xlabel("Quant", fontsize=12, fontweight="bold")
    ax.set_ylabel("Throughput (toks/s)", fontsize=12, fontweight="bold")
    ax.set_title("Throughput", fontsize=14, fontweight="bold")
    ax.legend(loc="best", frameon=True)
    _style_axes(ax, y_only=True)
    return _finish_plot(df, output_path, "throughput.png")


def plot_latency(
    results: List[EvaluationResult],
    output_path: Optional[Path] = None,
) -> Path:
    """Bar chart of mean decode latency (ms/tok) from llama-bench."""
    console.print("[cyan]Creating latency plot...[/cyan]")

    rows = []
    for r in results:
        ms = r.throughput_latency_ms
        if ms is None and r.throughput_tokens_per_sec:
            ms = 1000.0 / r.throughput_tokens_per_sec
        if ms is None:
            continue
        rows.append({
            "model": _pretty_model(r.model_name),
            "quant": _pretty_quant(r.quantization_type),
            "latency_ms": ms,
        })
    if not rows:
        console.print("[yellow]No latency data available for plotting[/yellow]")
        return None

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 7))
    unique_models = list(df["model"].unique())
    colors = _model_colors(unique_models)
    quants = list(dict.fromkeys(df["quant"]))
    width = 0.8 / max(len(unique_models), 1)
    x = range(len(quants))
    for i, model in enumerate(unique_models):
        model_df = df[df["model"] == model].set_index("quant")
        heights = [float(model_df.loc[q, "latency_ms"]) if q in model_df.index else 0 for q in quants]
        offset = width * (i - (len(unique_models) - 1) / 2)
        ax.bar(
            [j + offset for j in x],
            heights,
            width,
            label=model,
            color=colors[model],
            edgecolor=_BAR_EDGE,
            linewidth=0.7,
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(quants, rotation=30, ha="right")
    ax.set_xlabel("Quant", fontsize=12, fontweight="bold")
    ax.set_ylabel("ms/tok", fontsize=12, fontweight="bold")
    ax.set_title("Latency", fontsize=14, fontweight="bold")
    ax.legend(loc="best", frameon=True)
    _style_axes(ax, y_only=True)
    return _finish_plot(df, output_path, "latency.png")


def plot_compression_ratio(
    results: List[EvaluationResult],
    output_path: Optional[Path] = None,
) -> Path:
    """Bar chart of compression ratio vs the FP16 (or largest) baseline."""
    console.print("[cyan]Creating compression ratio plot...[/cyan]")

    rows = []
    for r in results:
        ratio = r.compression_ratio
        if ratio is None and r.file_size_gb:
            # Fallback if JSON predates annotate_relative_metrics
            family = [x for x in results if x.model_name == r.model_name]
            baseline_size = max((x.file_size_gb or 0) for x in family) or None
            if baseline_size:
                ratio = baseline_size / r.file_size_gb
        if ratio is None:
            continue
        rows.append({
            "model": _pretty_model(r.model_name),
            "quant": _pretty_quant(r.quantization_type),
            "ratio": ratio,
            "bpw": r.bits_per_weight,
        })
    if not rows:
        console.print("[yellow]No compression ratio data available for plotting[/yellow]")
        return None

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 7))
    unique_models = list(df["model"].unique())
    quants = list(dict.fromkeys(df["quant"]))
    quant_colors = dict(zip(quants, _palette(len(quants))))
    width = 0.8 / max(len(quants), 1)
    x = range(len(unique_models))
    for i, quant in enumerate(quants):
        qdf = df[df["quant"] == quant].set_index("model")
        heights = [float(qdf.loc[m, "ratio"]) if m in qdf.index else 0 for m in unique_models]
        offset = width * (i - (len(quants) - 1) / 2)
        bars = ax.bar(
            [j + offset for j in x],
            heights,
            width,
            label=quant,
            color=quant_colors[quant],
            edgecolor=_BAR_EDGE,
            linewidth=0.7,
        )
        for bar, model in zip(bars, unique_models):
            if model in qdf.index and pd.notna(qdf.loc[model, "bpw"]):
                ax.annotate(
                    f"{qdf.loc[model, 'bpw']:.1f} bpw",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                    color=quant_colors[quant],
                )

    ax.axhline(1.0, color="#b3b3b3", linestyle="--", linewidth=1.1, alpha=0.9, label="Baseline")
    ax.set_xticks(list(x))
    ax.set_xticklabels(unique_models)
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Ratio (×)", fontsize=12, fontweight="bold")
    ax.set_title("Compression", fontsize=14, fontweight="bold")
    ax.legend(loc="best", frameon=True, fontsize=8)
    _style_axes(ax, y_only=True)
    return _finish_plot(df, output_path, "compression.png")


def _finish_plot(df: pd.DataFrame, output_path: Optional[Path], default_name: str) -> Path:
    """Write PNG + matching CSV into report/ (or output_path's directory)."""
    if output_path is None:
        output_path = CHARTS_DIR / default_name
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=CHART_DPI, bbox_inches="tight")
    plt.close()
    console.print(f"[green]✓ Saved to {output_path}[/green]")
    console.print(f"[green]✓ Saved to {csv_path}[/green]")
    return output_path


def plot_perplexity_vs_throughput(
    results: List[EvaluationResult],
    output_path: Optional[Path] = None,
) -> Path:
    """Perplexity (y) vs decode throughput (x)."""
    console.print("[cyan]Creating perplexity vs throughput plot...[/cyan]")
    unique_models = list(dict.fromkeys(_pretty_model(r.model_name) for r in results))
    styles = _series_for(unique_models)
    fig, ax = plt.subplots(figsize=(12, 7))
    rows = []
    drew = False
    for model in unique_models:
        points = []
        for r in results:
            if _pretty_model(r.model_name) != model:
                continue
            if r.perplexity is None or r.throughput_tokens_per_sec is None:
                continue
            points.append((
                r.throughput_tokens_per_sec,
                r.perplexity,
                _pretty_quant(r.quantization_type),
            ))
        if not points:
            continue
        points.sort(key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        st = styles[model]
        ax.plot(
            xs,
            ys,
            color=st["color"],
            linestyle=st["ls"],
            marker=st["marker"],
            markersize=8,
            markeredgecolor=st["color"],
            markerfacecolor=st["color"],
            markeredgewidth=1.2,
            linewidth=1.8,
            label=model,
        )
        for x, y, lab in points:
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, 6), fontsize=8, color=_ANNOTATE)
            rows.append({
                "model": model,
                "quant": lab,
                "tok_s": x,
                "perplexity": y,
            })
        drew = True
    if not drew:
        plt.close()
        console.print("[yellow]No perplexity/throughput data available for plotting[/yellow]")
        return None
    ax.set_xlabel("Throughput (toks/s)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Perplexity", fontsize=12, fontweight="bold")
    ax.set_title("Perplexity vs Throughput", fontsize=14, fontweight="bold")
    ax.legend(frameon=True)
    _style_axes(ax)
    ax.invert_yaxis()
    return _finish_plot(pd.DataFrame(rows), output_path, "perplexity_vs_throughput.png")


def plot_accuracy_vs_size(
    results: List[EvaluationResult],
    output_path: Optional[Path] = None,
) -> Path:
    """Downstream accuracy vs model size (one series per task)."""
    console.print("[cyan]Creating accuracy vs size plot...[/cyan]")
    df = _accuracy_frame(results)
    if df is None or df["size_gb"].isna().all():
        console.print("[yellow]No benchmark scores available for plotting[/yellow]")
        return None

    fig, ax = plt.subplots(figsize=(12, 7))
    _draw_acc_series(ax, df, "size_gb", "acc", annotate=True)
    ax.set_xlabel("Size (GB)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("Accuracy vs Size", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9, frameon=True)
    _style_axes(ax)
    return _finish_plot(df[["model", "quant", "task", "acc", "size_gb"]], output_path, "accuracy.png")


def plot_accuracy_vs_precision(
    results: List[EvaluationResult],
    output_path: Optional[Path] = None,
) -> Path:
    """Accuracy vs GGUF precision (W16A16 … W2A16)."""
    console.print("[cyan]Creating accuracy vs precision plot...[/cyan]")
    df = _accuracy_frame(results)
    if df is None:
        console.print("[yellow]No benchmark scores available for plotting[/yellow]")
        return None

    order = sorted(df["quant"].unique(), key=_precision_rank)
    rank = {q: i for i, q in enumerate(order)}
    df = df.copy()
    df["x"] = df["quant"].map(rank)

    fig, ax = plt.subplots(figsize=(12, 7))
    _draw_acc_series(ax, df, "x", "acc", annotate=False)
    ax.set_xticks(list(range(len(order))))
    ax.set_xticklabels(order, rotation=0)
    ax.set_xlabel("Precision", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("Accuracy vs Precision", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9, frameon=True)
    _style_axes(ax, y_only=True)
    return _finish_plot(
        df[["model", "quant", "task", "acc"]],
        output_path,
        "accuracy_vs_precision.png",
    )


def plot_accuracy_drop(
    results: List[EvaluationResult],
    output_path: Optional[Path] = None,
) -> Path:
    """Accuracy drop vs FP16 (delta_acc) across precision."""
    console.print("[cyan]Creating accuracy drop vs FP16 plot...[/cyan]")
    df = _accuracy_frame(results, require_delta=True)
    if df is None:
        console.print("[yellow]No accuracy-drop (delta_acc) data available for plotting[/yellow]")
        return None

    order = sorted(df["quant"].unique(), key=_precision_rank)
    rank = {q: i for i, q in enumerate(order)}
    df = df.copy()
    df["x"] = df["quant"].map(rank)

    fig, ax = plt.subplots(figsize=(12, 7))
    _draw_acc_series(ax, df, "x", "delta_acc", annotate=False)
    ax.axhline(0.0, color="#b3b3b3", linestyle="--", linewidth=1.1, alpha=0.9, zorder=0)
    ax.set_xticks(list(range(len(order))))
    ax.set_xticklabels(order)
    ax.set_xlabel("Precision", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy drop vs FP16", fontsize=12, fontweight="bold")
    ax.set_title("Accuracy Drop vs FP16", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9, frameon=True)
    _style_axes(ax, y_only=True)
    return _finish_plot(
        df[["model", "quant", "task", "delta_acc"]],
        output_path,
        "accuracy_drop.png",
    )


def plot_accuracy_vs_throughput(
    results: List[EvaluationResult],
    output_path: Optional[Path] = None,
) -> Path:
    """Accuracy (y) vs decode throughput (x)."""
    console.print("[cyan]Creating accuracy vs throughput plot...[/cyan]")
    df = _accuracy_frame(results, require_throughput=True)
    if df is None:
        console.print("[yellow]No accuracy/throughput data available for plotting[/yellow]")
        return None

    fig, ax = plt.subplots(figsize=(12, 7))
    _draw_acc_series(ax, df, "tok_s", "acc", annotate=True)
    ax.set_xlabel("Throughput (toks/s)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("Accuracy vs Throughput", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9, frameon=True)
    _style_axes(ax)
    return _finish_plot(
        df[["model", "quant", "task", "acc", "tok_s"]],
        output_path,
        "accuracy_vs_throughput.png",
    )


def generate_all_charts(
    results_file: Path,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Generate one PNG and matching CSV per chart into report/ (or output_dir)."""
    console.print(f"\n[bold]Generating charts from {results_file}...[/bold]\n")

    results = load_results(results_file)
    if not results:
        console.print("[red]No results to visualize[/red]")
        return {}

    chart_dir = Path(output_dir) if output_dir else CHARTS_DIR
    chart_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("perplexity_vs_size", plot_size_vs_quality, "perplexity_vs_size.png"),
        ("throughput", plot_throughput_comparison, "throughput.png"),
        ("latency", plot_latency, "latency.png"),
        ("compression", plot_compression_ratio, "compression.png"),
        ("perplexity_vs_throughput", plot_perplexity_vs_throughput, "perplexity_vs_throughput.png"),
        ("accuracy", plot_accuracy_vs_size, "accuracy.png"),
        ("accuracy_vs_precision", plot_accuracy_vs_precision, "accuracy_vs_precision.png"),
        ("accuracy_drop", plot_accuracy_drop, "accuracy_drop.png"),
        ("accuracy_vs_throughput", plot_accuracy_vs_throughput, "accuracy_vs_throughput.png"),
    ]
    chart_paths = {}
    for name, fn, filename in jobs:
        path = fn(results, chart_dir / filename)
        if path:
            chart_paths[name] = path

    console.print(f"\n[bold green]Generated {len(chart_paths)} chart(s)[/bold green]")
    return chart_paths

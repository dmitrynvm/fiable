"""Visualization functionality for compression analysis."""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rich.console import Console

from fiable.utils import helpers
from fiable.config.settings import (
    STORE_DIR,
    CHARTS_DIR,
    CHART_DPI,
    CHART_COLORS,
    MODELS,
    DEFAULT_QUANT_TYPES,
    parse_quant_spec,
    llama_src_dir,
)
from fiable.core.evaluate import EvaluationResult, summary_row
from fiable.core.metrics import (
    QUANT_ALIASES,
    annotate_relative_metrics,
    is_baseline_quant,
    pick_baseline,
    read_gguf_parameter_count,
)


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
    {"color": "#d62728", "ls": "-", "marker": "s"},
    {"color": "#2ca02c", "ls": "-", "marker": "D"},
    {"color": "#ff7f0e", "ls": "-", "marker": "^"},
    {"color": "#9467bd", "ls": "-", "marker": "v"},
    {"color": "#8c564b", "ls": "-", "marker": "P"},
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


# Public comparison methods (baselines first).
_QUANT_ORDER = (
    "FP32",
    "F32",
    "FP16",
    "F16",
    "BF16",
    "Q4_K_M",
    "GPTQ_4",
    "EVOPRESS_4",
)


def _pretty_quant(quant: str) -> str:
    """Canonical GGUF method name: F16 → FP16, W4A16 → Q4_K_M, Q4_K_M unchanged."""
    if not quant:
        return quant
    raw = str(quant).strip()
    upper = raw.upper().replace(" ", "")
    if upper in QUANT_ALIASES:
        return QUANT_ALIASES[upper]
    for name in _QUANT_ORDER:
        if name.upper() == upper:
            return name
    for name in DEFAULT_QUANT_TYPES:
        if str(name).upper() == upper:
            return name
    try:
        mapped = parse_quant_spec(raw)
    except ValueError:
        return raw
    mapped_u = str(mapped).upper()
    if mapped_u in QUANT_ALIASES:
        return QUANT_ALIASES[mapped_u]
    for name in _QUANT_ORDER:
        if name.upper() == mapped_u:
            return name
    return mapped


def _quant_rank(label: str) -> tuple:
    pretty = _pretty_quant(label)
    key = pretty.upper()
    order = [name.upper() for name in _QUANT_ORDER]
    if key in order:
        return (order.index(key), pretty)
    return (len(_QUANT_ORDER), pretty)


def _ordered_quants(labels) -> List[str]:
    return sorted(dict.fromkeys(labels), key=_quant_rank)


_TASK_ORDER = ("mmlu", "gsm8k", "humaneval")
_TASK_MARKERS = ("o", "s", "^", "D", "v", "P")
_TASK_LINESTYLES = ("-", "--", ":", "-.", "-", "--")


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
            retention = _finite((r.acc_retention or {}).get(task))
            tok_s = _finite(r.throughput_tokens_per_sec)
            if require_throughput and tok_s is None:
                continue
            rows.append({
                "model": _pretty_model(r.model_name),
                "quant": _pretty_quant(r.quantization_type),
                "task": task,
                "acc": acc,
                "delta_acc": delta,
                "acc_retention": retention,
                "size_gb": r.file_size_gb,
                "tok_s": tok_s,
            })
    if not rows:
        return None
    return pd.DataFrame(rows)


_ACC_EXPLORATORY = (
    "exploratory: lm-eval --limit 50, not a full-task ranking"
)


def _eval_frame(results: List[EvaluationResult]) -> pd.DataFrame:
    """One row per GGUF with the metrics used by research plots."""
    rows = []
    for r in results:
        delta = _finite(r.delta_ppl)
        ppl_ret = None
        if delta is not None:
            ppl_ret = 1.0 / (1.0 + delta) if delta > -0.999 else None
        eff = None
        if r.efficiency_score:
            for task in _TASK_ORDER:
                if task in r.efficiency_score:
                    eff = _finite(r.efficiency_score[task])
                    break
            if eff is None:
                eff = _finite(next(iter(r.efficiency_score.values()), None))
        rows.append({
            "model": _pretty_model(r.model_name),
            "quant": _pretty_quant(r.quantization_type),
            "bits": _finite(r.bits_per_weight),
            "size_gb": _finite(r.file_size_gb),
            "size_reduction": _finite(r.size_reduction),
            "delta_ppl": delta,
            "ppl_retention": ppl_ret,
            "perplexity": _finite(r.perplexity),
            "perplexity_long": _finite(r.perplexity_long),
            "kl": _finite(r.kl_divergence),
            "top1": _finite(r.top1_match),
            "decode_tok_s": _finite(r.throughput_tokens_per_sec),
            "prefill_tok_s": _finite(r.prefill_tokens_per_sec),
            "speedup": _finite(r.speedup),
            "prefill_speedup": _finite(r.prefill_speedup),
            "vram_gb": _finite(r.peak_vram_gb) or _finite(r.throughput_memory_gb),
            "efficiency": eff,
        })
    return pd.DataFrame(rows)


def _draw_model_lines(
    ax,
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    *,
    annotate: bool = True,
    sort: bool = True,
) -> None:
    styles = _series_for(list(dict.fromkeys(df["model"])))
    for model in dict.fromkeys(df["model"]):
        part = df[df["model"] == model].dropna(subset=[xcol, ycol]).copy()
        if part.empty:
            continue
        if sort:
            part = part.sort_values(xcol)
        st = styles[model]
        ax.plot(
            part[xcol],
            part[ycol],
            marker=st["marker"],
            markersize=8,
            markeredgecolor=st["color"],
            markerfacecolor=st["color"],
            markeredgewidth=1.2,
            label=model,
            color=st["color"],
            linestyle=st["ls"],
            linewidth=1.8,
            zorder=3,
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


def _pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """True for points not dominated when maximizing both x and y."""
    keep = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        if not keep[i]:
            continue
        better = (x >= x[i]) & (y >= y[i]) & ((x > x[i]) | (y > y[i]))
        if np.any(better):
            keep[i] = False
    return keep


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
                marker=st["marker"],
                markersize=8,
                markeredgecolor=st["color"],
                markerfacecolor=st["color"],
                markeredgewidth=1.2,
                label=f"{model} {task}",
                color=st["color"],
                linestyle=ts["ls"],
                linewidth=1.8,
                zorder=3,
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
    ax.set_title(f"Accuracy vs Size ({_ACC_EXPLORATORY})", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9, frameon=True)
    _style_axes(ax)
    return _finish_plot(df[["model", "quant", "task", "acc", "size_gb"]], output_path, "accuracy_vs_size.png")


def plot_accuracy_drop(
    results: List[EvaluationResult],
    output_path: Optional[Path] = None,
) -> Path:
    """Accuracy drop vs FP16 (delta_acc) across quantization methods."""
    console.print("[cyan]Creating accuracy drop vs FP16 plot...[/cyan]")
    df = _accuracy_frame(results, require_delta=True)
    if df is None:
        console.print("[yellow]No accuracy-drop (delta_acc) data available for plotting[/yellow]")
        return None

    # FP16 is the baseline (drop is always 0); plot quantized methods only.
    df = df[~df["quant"].map(is_baseline_quant)].copy()
    if df.empty:
        console.print("[yellow]No quantized accuracy-drop points to plot[/yellow]")
        return None

    order = sorted(df["quant"].unique(), key=_quant_rank)
    rank = {q: i for i, q in enumerate(order)}
    df = df.copy()
    df["x"] = df["quant"].map(rank)

    fig, ax = plt.subplots(figsize=(12, 7))
    _draw_acc_series(ax, df, "x", "delta_acc", annotate=False)
    ax.axhline(0.0, color="#b3b3b3", linestyle="--", linewidth=1.1, alpha=0.9, zorder=0)
    ax.set_xticks(list(range(len(order))))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_xlabel("Quantization method", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy drop vs FP16", fontsize=12, fontweight="bold")
    ax.set_title(f"Accuracy Drop vs FP16 ({_ACC_EXPLORATORY})", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9, frameon=True)
    _style_axes(ax, y_only=True)
    return _finish_plot(
        df[["model", "quant", "task", "delta_acc"]],
        output_path,
        "accuracy_drop.png",
    )


def plot_perplexity_drop(
    results: List[EvaluationResult],
    output_path: Optional[Path] = None,
) -> Path:
    """Relative PPL drop vs FP16 (delta_ppl) across quantization methods."""
    console.print("[cyan]Creating perplexity drop vs FP16 plot...[/cyan]")
    df = _eval_frame(results).dropna(subset=["delta_ppl"])
    df = df[~df["quant"].map(is_baseline_quant)].copy()
    if df.empty:
        console.print("[yellow]No quantized perplexity-drop points to plot[/yellow]")
        return None

    order = sorted(df["quant"].unique(), key=_quant_rank)
    rank = {q: i for i, q in enumerate(order)}
    df["x"] = df["quant"].map(rank)

    fig, ax = plt.subplots(figsize=(12, 7))
    _draw_model_lines(ax, df, "x", "delta_ppl", annotate=False, sort=True)
    ax.axhline(0.0, color="#b3b3b3", linestyle="--", linewidth=1.1, alpha=0.9, zorder=0)
    ax.set_xticks(list(range(len(order))))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_xlabel("Quantization method", fontsize=12, fontweight="bold")
    ax.set_ylabel("Relative PPL vs FP16 (ΔPPL)", fontsize=12, fontweight="bold")
    ax.set_title("Perplexity Drop vs FP16", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10, frameon=True)
    _style_axes(ax, y_only=True)
    return _finish_plot(
        df[["model", "quant", "delta_ppl"]],
        output_path,
        "perplexity_drop.png",
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
    ax.set_title(f"Accuracy vs Throughput ({_ACC_EXPLORATORY})", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9, frameon=True)
    _style_axes(ax)
    return _finish_plot(
        df[["model", "quant", "task", "acc", "tok_s"]],
        output_path,
        "accuracy_vs_throughput.png",
    )


def plot_delta_ppl_vs_bits(
    results: List[EvaluationResult],
    output_path: Optional[Path] = None,
) -> Path:
    """Relative WikiText PPL vs bits/weight (0 = FP16)."""
    console.print("[cyan]Creating relative PPL vs bits plot...[/cyan]")
    df = _eval_frame(results).dropna(subset=["bits", "delta_ppl"])
    if df.empty:
        console.print("[yellow]No delta_ppl / bits_per_weight data available for plotting[/yellow]")
        return None

    fig, ax = plt.subplots(figsize=(12, 7))
    _draw_model_lines(ax, df, "bits", "delta_ppl")
    ax.axhline(0.0, color="#b3b3b3", linestyle="--", linewidth=1.1, alpha=0.9, zorder=0)
    ax.set_xlabel("Bits per weight", fontsize=12, fontweight="bold")
    ax.set_ylabel("Relative PPL vs FP16 (ΔPPL)", fontsize=12, fontweight="bold")
    ax.set_title("Relative perplexity vs bits per weight", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10, frameon=True)
    _style_axes(ax)
    return _finish_plot(
        df[["model", "quant", "bits", "delta_ppl"]],
        output_path,
        "delta_ppl_vs_bits.png",
    )


def plot_kl_top1_vs_bits(
    results: List[EvaluationResult],
    output_path: Optional[Path] = None,
) -> Path:
    """KL(baseline ‖ quant) and top-1 match vs bits/weight."""
    console.print("[cyan]Creating KL / top-1 vs bits plot...[/cyan]")
    df = _eval_frame(results)
    kl_df = df.dropna(subset=["bits", "kl"])
    top_df = df.dropna(subset=["bits", "top1"])
    if kl_df.empty and top_df.empty:
        console.print("[yellow]No KL / top-1 data available for plotting[/yellow]")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if not kl_df.empty:
        _draw_model_lines(axes[0], kl_df, "bits", "kl")
    axes[0].set_xlabel("Bits per weight", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("KL(baseline ‖ quant)", fontsize=12, fontweight="bold")
    axes[0].set_title("Distributional drift", fontsize=13, fontweight="bold")
    axes[0].legend(loc="best", fontsize=9, frameon=True)
    _style_axes(axes[0])

    if not top_df.empty:
        _draw_model_lines(axes[1], top_df, "bits", "top1")
    axes[1].axhline(1.0, color="#b3b3b3", linestyle="--", linewidth=1.1, alpha=0.9, zorder=0)
    axes[1].set_xlabel("Bits per weight", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Top-1 token match vs FP16", fontsize=12, fontweight="bold")
    axes[1].set_title("Next-token agreement", fontsize=13, fontweight="bold")
    axes[1].legend(loc="best", fontsize=9, frameon=True)
    _style_axes(axes[1])
    fig.suptitle("Distributional fidelity vs bits per weight", fontsize=14, fontweight="bold")
    csv = df[["model", "quant", "bits", "kl", "top1"]].dropna(how="all", subset=["kl", "top1"])
    return _finish_plot(csv, output_path, "kl_top1_vs_bits.png")


def plot_prefill_vs_decode(
    results: List[EvaluationResult],
    output_path: Optional[Path] = None,
) -> Path:
    """Decode speedup vs prefill speedup by bits (quant kernel story)."""
    console.print("[cyan]Creating prefill vs decode speedup plot...[/cyan]")
    df = _eval_frame(results)
    dec = df.dropna(subset=["bits", "speedup"])
    pref = df.dropna(subset=["bits", "prefill_speedup"])
    if dec.empty and pref.empty:
        console.print("[yellow]No prefill / decode speedup data available for plotting[/yellow]")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    if not dec.empty:
        _draw_model_lines(axes[0], dec, "bits", "speedup")
    axes[0].axhline(1.0, color="#b3b3b3", linestyle="--", linewidth=1.1, alpha=0.9, zorder=0)
    axes[0].set_xlabel("Bits per weight", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Decode speedup vs FP16", fontsize=12, fontweight="bold")
    axes[0].set_title("Token generation", fontsize=13, fontweight="bold")
    axes[0].legend(loc="best", fontsize=9, frameon=True)
    _style_axes(axes[0])

    if not pref.empty:
        _draw_model_lines(axes[1], pref, "bits", "prefill_speedup")
    axes[1].axhline(1.0, color="#b3b3b3", linestyle="--", linewidth=1.1, alpha=0.9, zorder=0)
    axes[1].set_xlabel("Bits per weight", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Prefill speedup vs FP16", fontsize=12, fontweight="bold")
    axes[1].set_title("Prompt processing", fontsize=13, fontweight="bold")
    axes[1].legend(loc="best", fontsize=9, frameon=True)
    _style_axes(axes[1])
    fig.suptitle("Prefill vs decode speedup", fontsize=14, fontweight="bold")
    csv = df[["model", "quant", "bits", "speedup", "prefill_speedup", "decode_tok_s", "prefill_tok_s"]]
    csv = csv.dropna(how="all", subset=["speedup", "prefill_speedup"])
    return _finish_plot(csv, output_path, "prefill_vs_decode.png")


def plot_perplexity_short_vs_long(
    results: List[EvaluationResult],
    output_path: Optional[Path] = None,
) -> Path:
    """WikiText PPL at ctx 512 vs 2048, grouped by quantization method."""
    console.print("[cyan]Creating short vs long context PPL plot...[/cyan]")
    df = _eval_frame(results).dropna(subset=["perplexity", "perplexity_long"])
    if df.empty:
        console.print("[yellow]No short/long perplexity data available for plotting[/yellow]")
        return None

    models = list(dict.fromkeys(df["model"]))
    n = max(len(models), 1)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), sharey=False)
    if n == 1:
        axes = [axes]
    csv_rows = []
    for ax, model in zip(axes, models):
        part = df[df["model"] == model].copy()
        part["rank"] = part["quant"].map(_quant_rank)
        part = part.sort_values("rank")
        xs = np.arange(len(part))
        width = 0.38
        ax.bar(
            xs - width / 2,
            part["perplexity"],
            width,
            label="ctx 512",
            color="#1f77b4",
            edgecolor=_BAR_EDGE,
            linewidth=0.6,
        )
        ax.bar(
            xs + width / 2,
            part["perplexity_long"],
            width,
            label="ctx 2048",
            color="#ff7f0e",
            edgecolor=_BAR_EDGE,
            linewidth=0.6,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(list(part["quant"]), rotation=30, ha="right")
        ax.set_title(model, fontsize=13, fontweight="bold")
        ax.set_xlabel("Quantization method", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9, frameon=True)
        _style_axes(ax, y_only=True)
        for _, row in part.iterrows():
            csv_rows.append({
                "model": model,
                "quant": row["quant"],
                "perplexity": row["perplexity"],
                "perplexity_long": row["perplexity_long"],
            })
    axes[0].set_ylabel("Perplexity (lower is better)", fontsize=12, fontweight="bold")
    fig.suptitle("WikiText perplexity at context 512 vs 2048", fontsize=14, fontweight="bold")
    return _finish_plot(pd.DataFrame(csv_rows), output_path, "perplexity_short_vs_long.png")


def plot_efficiency_pareto(
    results: List[EvaluationResult],
    output_path: Optional[Path] = None,
) -> Path:
    """Size reduction vs WikiText PPL retention; dashed Pareto front per model."""
    console.print("[cyan]Creating size–quality Pareto plot...[/cyan]")
    df = _eval_frame(results).dropna(subset=["size_reduction", "ppl_retention"])
    df = df[df["size_reduction"] > 0].copy()
    if df.empty:
        console.print("[yellow]No size-reduction / PPL retention data available for plotting[/yellow]")
        return None

    fig, ax = plt.subplots(figsize=(12, 7))
    styles = _series_for(list(dict.fromkeys(df["model"])))
    csv_rows = []
    for model in dict.fromkeys(df["model"]):
        part = df[df["model"] == model].reset_index(drop=True)
        st = styles[model]
        ax.scatter(
            part["size_reduction"],
            part["ppl_retention"],
            s=70,
            color=st["color"],
            marker=st["marker"],
            edgecolors="#222222",
            linewidths=0.6,
            zorder=3,
            label=model,
        )
        x = part["size_reduction"].to_numpy(dtype=float)
        y = part["ppl_retention"].to_numpy(dtype=float)
        mask = _pareto_mask(x, y)
        front = part.loc[mask].sort_values("size_reduction")
        if len(front) >= 2:
            ax.plot(
                front["size_reduction"],
                front["ppl_retention"],
                color=st["color"],
                linestyle="--",
                linewidth=1.4,
                zorder=2,
            )
        for i, row in part.iterrows():
            ax.annotate(
                row["quant"],
                (row["size_reduction"], row["ppl_retention"]),
                fontsize=8,
                xytext=(4, 6),
                textcoords="offset points",
                color=_ANNOTATE,
            )
            csv_rows.append({
                "model": model,
                "quant": row["quant"],
                "size_reduction": row["size_reduction"],
                "ppl_retention": row["ppl_retention"],
                "delta_ppl": row["delta_ppl"],
                "efficiency": row["efficiency"],
                "on_pareto": bool(mask[i]),
            })

    ax.set_xlabel("Size reduction vs FP16", fontsize=12, fontweight="bold")
    ax.set_ylabel("PPL retention (baseline PPL / PPL)", fontsize=12, fontweight="bold")
    ax.set_title("Compression–quality Pareto (WikiText)", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10, frameon=True)
    _style_axes(ax)
    return _finish_plot(pd.DataFrame(csv_rows), output_path, "efficiency_pareto.png")


_WEIGHT_SAMPLE_N = 400_000
_WEIGHT_BINS = 256
_WEIGHT_UNIQUE_ROUND = 1e-6
_WEIGHT_RUG_MAX = 400
_WEIGHT_MAX_TENSORS = 16
_SKIP_TENSOR_SUBSTR = ("norm", "bias", "mask", "rope", "embd", "embed")


def _import_gguf():
    """Load llama.cpp gguf-py (store clone or already installed)."""
    try:
        import gguf
        from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
        from gguf.quants import dequantize
        return gguf, dequantize, GGMLQuantizationType, GGML_QUANT_SIZES
    except ImportError:
        src = llama_src_dir() / "gguf-py"
        if src.is_dir() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
        import gguf
        from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
        from gguf.quants import dequantize
        return gguf, dequantize, GGMLQuantizationType, GGML_QUANT_SIZES


def _keep_weight_tensor(tensor) -> bool:
    name = str(tensor.name).lower()
    if any(token in name for token in _SKIP_TENSOR_SUBSTR):
        return False
    if name.endswith("output.weight"):
        return False
    return int(tensor.n_elements) >= 4096


def _sample_tensor_weights(tensor, n_take: int, rng, dequantize, qtype_enum, quant_sizes) -> np.ndarray:
    if n_take <= 0:
        return np.empty(0, dtype=np.float32)
    qtype = tensor.tensor_type
    data = np.asarray(tensor.data)
    float_types = {qtype_enum.F16, qtype_enum.F32, qtype_enum.F64, qtype_enum.BF16}
    if qtype in float_types:
        try:
            flat = dequantize(data, qtype).astype(np.float32, copy=False).ravel()
        except Exception:
            flat = np.asarray(data, dtype=np.float32).ravel()
        if flat.size <= n_take:
            return np.ascontiguousarray(flat)
        return rng.choice(flat, size=n_take, replace=False).astype(np.float32, copy=False)

    rows = data.reshape(-1, data.shape[-1])
    block_size, type_size = quant_sizes[qtype]
    if type_size <= 0:
        return np.empty(0, dtype=np.float32)
    elems_per_row = (rows.shape[-1] // type_size) * block_size
    if elems_per_row <= 0 or rows.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    n_rows_needed = min(rows.shape[0], max(1, int(np.ceil(n_take / elems_per_row))))
    idx = rng.choice(rows.shape[0], size=n_rows_needed, replace=False)
    weights = dequantize(np.ascontiguousarray(rows[idx]), qtype)
    weights = np.asarray(weights, dtype=np.float32).ravel()
    if weights.size > n_take:
        weights = rng.choice(weights, size=n_take, replace=False)
    return weights.astype(np.float32, copy=False)


def sample_gguf_weights(path: Path, n: int = _WEIGHT_SAMPLE_N, seed: int = 0) -> np.ndarray:
    """Dequantize a random subset of 2D layer weights from a GGUF."""
    gguf, dequantize, qtype_enum, quant_sizes = _import_gguf()
    reader = gguf.GGUFReader(str(path))
    rng = np.random.default_rng(seed)
    tensors = [t for t in reader.tensors if _keep_weight_tensor(t)]
    if not tensors:
        tensors = [t for t in reader.tensors if int(t.n_elements) >= 4096]
    tensors = sorted(tensors, key=lambda t: int(t.n_elements), reverse=True)[:_WEIGHT_MAX_TENSORS]
    rng.shuffle(tensors)
    parts: List[np.ndarray] = []
    left = int(n)
    for i, tensor in enumerate(tensors):
        if left <= 0:
            break
        remaining_t = len(tensors) - i
        take = min(left, max(left // remaining_t, min(left, 40_000)))
        try:
            sample = _sample_tensor_weights(
                tensor, take, rng, dequantize, qtype_enum, quant_sizes
            )
        except Exception as exc:
            console.print(f"[dim]    skip {tensor.name}: {exc}[/dim]")
            continue
        if sample.size:
            parts.append(sample)
            left -= int(sample.size)
    if not parts:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(parts)


def plot_weight_distributions(
    results: List[EvaluationResult],
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """Histogram of dequantized weights, one panel per quantization method."""
    console.print("[cyan]Creating weight distribution plot...[/cyan]")
    grouped: Dict[str, List[EvaluationResult]] = defaultdict(list)
    for result in results:
        path = Path(result.model_path)
        if path.exists() and path.suffix.lower() == ".gguf":
            grouped[result.model_name].append(result)
    if not grouped:
        console.print("[yellow]No GGUF files found for weight distributions[/yellow]")
        return None

    try:
        _import_gguf()
    except ImportError:
        console.print(
            "[yellow]gguf-py not found; skip weight distributions "
            "(need store/llama.cpp/gguf-py)[/yellow]"
        )
        return None

    samples: Dict[Tuple[str, str], np.ndarray] = {}
    model_names = []
    quant_labels = []
    for model_name, group in grouped.items():
        pretty_model = _pretty_model(model_name)
        model_names.append(pretty_model)
        ordered = sorted(
            group, key=lambda r: _quant_rank(_pretty_quant(r.quantization_type))
        )
        for result in ordered:
            quant = _pretty_quant(result.quantization_type)
            path = Path(result.model_path)
            console.print(f"[dim]  sampling {path.name}...[/dim]")
            arr = sample_gguf_weights(path)
            if arr.size == 0:
                console.print(f"[yellow]  no weights sampled from {path.name}[/yellow]")
                continue
            samples[(pretty_model, quant)] = arr
            if quant not in quant_labels:
                quant_labels.append(quant)

    if not samples:
        console.print("[yellow]No weight samples to plot[/yellow]")
        return None

    quant_labels = sorted(quant_labels, key=_quant_rank)
    model_names = list(dict.fromkeys(model_names))
    n_models, n_quants = len(model_names), len(quant_labels)
    fig, axes = plt.subplots(
        n_models,
        n_quants,
        figsize=(max(3.2 * n_quants, 8), max(3.0 * n_models, 4.5)),
        squeeze=False,
        sharex="row",
    )
    colors = _palette(n_quants)
    csv_rows = []

    for i, model in enumerate(model_names):
        ref = None
        for candidate in quant_labels:
            if (model, candidate) in samples and is_baseline_quant(candidate):
                ref = samples[(model, candidate)]
                break
        if ref is None:
            ref = next(samples[(model, q)] for q in quant_labels if (model, q) in samples)
        span = float(max(abs(np.percentile(ref, 0.5)), abs(np.percentile(ref, 99.5)), 1e-3))
        xlim = (-span, span)
        bins = np.linspace(xlim[0], xlim[1], _WEIGHT_BINS + 1)
        ref_density, _ = np.histogram(ref, bins=bins, density=True)
        ref_centers = 0.5 * (bins[:-1] + bins[1:])
        y_cap = 1.15 * float(np.max(ref_density)) if np.max(ref_density) > 0 else 1.0
        for j, quant in enumerate(quant_labels):
            ax = axes[i, j]
            arr = samples.get((model, quant))
            if arr is None:
                ax.axis("off")
                continue
            density, edges = np.histogram(arr, bins=bins, density=True)
            in_window = arr[(arr >= xlim[0]) & (arr <= xlim[1])]
            rounded = np.round(in_window / _WEIGHT_UNIQUE_ROUND) * _WEIGHT_UNIQUE_ROUND
            unique_vals = np.unique(rounded)
            n_unique = int(unique_vals.size)
            pct_zero = 100.0 * float(np.mean(np.abs(arr) < _WEIGHT_UNIQUE_ROUND))
            if not is_baseline_quant(quant):
                ax.fill_between(
                    ref_centers,
                    ref_density,
                    color="#9a9a9a",
                    alpha=0.35,
                    zorder=1,
                    step="mid",
                    label="FP16",
                )
            ax.hist(
                arr,
                bins=bins,
                density=True,
                histtype="stepfilled",
                color=colors[j],
                edgecolor="#222222",
                linewidth=0.6,
                alpha=0.45 if not is_baseline_quant(quant) else 0.72,
                zorder=2,
            )
            if not is_baseline_quant(quant) and unique_vals.size:
                rug = unique_vals
                if rug.size > _WEIGHT_RUG_MAX:
                    idx = np.linspace(0, rug.size - 1, _WEIGHT_RUG_MAX).astype(int)
                    rug = rug[idx]
                ax.plot(
                    rug,
                    np.zeros_like(rug),
                    "|",
                    ms=4,
                    color="#222222",
                    alpha=0.35,
                    zorder=4,
                    clip_on=False,
                )
            peak = float(np.max(density)) if density.size else 0.0
            ax.set_xlim(*xlim)
            ax.set_ylim(0, y_cap)
            if i == 0:
                ax.set_title(quant, fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{model}\nDensity", fontsize=9)
            if i == n_models - 1:
                ax.set_xlabel("Weight value", fontsize=9)
            notes = [f"σ={float(arr.std()):.3f}", f"n={n_unique}", f"%0={pct_zero:.1f}"]
            if peak > y_cap:
                notes.append(f"peak={peak:.0f}")
            ax.text(
                0.97,
                0.94,
                "\n".join(notes),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="#444444",
                linespacing=1.25,
            )
            _style_axes(ax, y_only=True)
            if i == 0 and j == 1:
                ax.legend(loc="upper left", fontsize=7, frameon=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            for center, dens in zip(centers, density):
                csv_rows.append(
                    {
                        "model": model,
                        "quant": quant,
                        "weight": float(center),
                        "density": float(dens),
                    }
                )

    fig.suptitle("Distribution of weights by quantization method", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _finish_plot(pd.DataFrame(csv_rows), output_path, "weight_distribution.png")


_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")
_LAYER_MAX_ROWS = 512
_KIND_SPECS = (
    ("attn_q.weight", "Q", "#1f77b4", "-"),
    ("attn_k.weight", "K", "#5fa8d3", "-"),
    ("attn_v.weight", "V", "#9ecae1", "-"),
    ("attn_output.weight", "O", "#9467bd", "-"),
    ("ffn_gate.weight", "gate", "#d62728", "--"),
    ("ffn_up.weight", "up", "#ff7f0e", "--"),
    ("ffn_down.weight", "down", "#2ca02c", "--"),
)
_KIND_LABEL = {name: label for name, label, _, _ in _KIND_SPECS}
_KIND_STYLE = {label: (color, ls) for _, label, color, ls in _KIND_SPECS}


def _packed_rows(tensor) -> np.ndarray:
    data = np.asarray(tensor.data)
    if data.ndim <= 1:
        return data.reshape(1, -1)
    return data.reshape(-1, data.shape[-1])


def _dequantize_array(tensor, dequantize) -> np.ndarray:
    data = np.ascontiguousarray(tensor.data)
    try:
        return np.asarray(dequantize(data, tensor.tensor_type), dtype=np.float32)
    except Exception:
        return np.asarray(data, dtype=np.float32)


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    ref = np.asarray(a, dtype=np.float32).ravel()
    est = np.asarray(b, dtype=np.float32).ravel()
    n = min(ref.size, est.size)
    if n <= 0:
        return float("nan")
    denom = float(np.linalg.norm(ref[:n]))
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(est[:n] - ref[:n]) / denom)


def tensor_rel_l2(fp_tensor, q_tensor, dequantize, qtype_enum, rng, max_rows: int = _LAYER_MAX_ROWS) -> float:
    """Relative L2 error ||Wq-W|| / ||W|| on a row subsample (exact if few rows)."""
    float_types = {qtype_enum.F16, qtype_enum.F32, qtype_enum.F64, qtype_enum.BF16}
    q_rows = _packed_rows(q_tensor)
    fp_data = np.asarray(fp_tensor.data)
    if (
        q_tensor.tensor_type not in float_types
        and fp_data.ndim >= 2
        and q_rows.shape[0] == fp_data.shape[0]
        and fp_tensor.tensor_type in float_types
    ):
        n_rows = int(q_rows.shape[0])
        take = min(int(max_rows), n_rows)
        # Sequential rows: mmap-friendly and within ~1e-4 of a random subsample.
        idx = np.arange(take)
        w_fp = np.asarray(fp_data[idx], dtype=np.float32)
        w_q = np.asarray(
            dequantize(np.ascontiguousarray(q_rows[idx]), q_tensor.tensor_type),
            dtype=np.float32,
        )
        return _rel_l2(w_fp, w_q)

    w_fp = _dequantize_array(fp_tensor, dequantize).ravel()
    w_q = _dequantize_array(q_tensor, dequantize).ravel()
    n = min(w_fp.size, w_q.size)
    cap = int(max_rows) * 256
    if n > cap:
        idx = rng.choice(n, size=cap, replace=False)
        return _rel_l2(w_fp[idx], w_q[idx])
    return _rel_l2(w_fp[:n], w_q[:n])


def layerwise_quant_error(
    fp_path: Path,
    quant_path: Path,
    seed: int = 0,
    fp_reader=None,
) -> List[dict]:
    """Per-block, per-tensor relative L2 error of a quantized GGUF vs its FP16 file."""
    gguf, dequantize, qtype_enum, _ = _import_gguf()
    close_fp = False
    if fp_reader is None:
        fp_reader = gguf.GGUFReader(str(fp_path))
        close_fp = True
    q_reader = gguf.GGUFReader(str(quant_path))
    fp_map = {t.name: t for t in fp_reader.tensors}
    rng = np.random.default_rng(seed)
    rows: List[dict] = []
    for tensor in q_reader.tensors:
        match = _BLK_RE.match(str(tensor.name))
        if not match:
            continue
        kind = match.group(2)
        if kind not in _KIND_LABEL:
            continue
        fp_tensor = fp_map.get(tensor.name)
        if fp_tensor is None:
            continue
        try:
            rel = tensor_rel_l2(fp_tensor, tensor, dequantize, qtype_enum, rng)
        except Exception as exc:
            console.print(f"[dim]    skip {tensor.name}: {exc}[/dim]")
            continue
        if rel != rel:
            continue
        rows.append(
            {
                "layer": int(match.group(1)),
                "tensor": _KIND_LABEL[kind],
                "rel_l2": rel,
            }
        )
    del q_reader
    if close_fp:
        del fp_reader
    return rows


def plot_layerwise_quant_error(
    results: List[EvaluationResult],
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """Relative L2 quantization error vs layer, one panel per model × method."""
    console.print("[cyan]Creating layer-wise quantization error plot...[/cyan]")
    grouped: Dict[str, List[EvaluationResult]] = defaultdict(list)
    for result in results:
        path = Path(result.model_path)
        if path.exists() and path.suffix.lower() == ".gguf":
            grouped[result.model_name].append(result)
    if not grouped:
        console.print("[yellow]No GGUF files found for layer-wise error[/yellow]")
        return None

    try:
        _import_gguf()
    except ImportError:
        console.print(
            "[yellow]gguf-py not found; skip layer-wise quantization error "
            "(need store/llama.cpp/gguf-py)[/yellow]"
        )
        return None

    series: Dict[Tuple[str, str], List[dict]] = {}
    model_names: List[str] = []
    quant_labels: List[str] = []
    for model_name, group in grouped.items():
        baseline = pick_baseline(group)
        if baseline is None or not is_baseline_quant(baseline.quantization_type):
            console.print(
                f"[yellow]  skip {model_name}: need FP16/BF16 GGUF as error baseline[/yellow]"
            )
            continue
        fp_path = Path(baseline.model_path)
        if not fp_path.exists():
            console.print(f"[yellow]  skip {model_name}: missing {fp_path}[/yellow]")
            continue
        pretty_model = _pretty_model(model_name)
        model_names.append(pretty_model)
        gguf, _, _, _ = _import_gguf()
        fp_reader = gguf.GGUFReader(str(fp_path))
        ordered = sorted(
            group, key=lambda r: _quant_rank(_pretty_quant(r.quantization_type))
        )
        for result in ordered:
            if is_baseline_quant(result.quantization_type):
                continue
            quant = _pretty_quant(result.quantization_type)
            path = Path(result.model_path)
            console.print(f"[dim]  error {path.name} vs {fp_path.name}...[/dim]")
            rows = layerwise_quant_error(fp_path, path, fp_reader=fp_reader)
            if not rows:
                console.print(f"[yellow]  no layer tensors matched in {path.name}[/yellow]")
                continue
            series[(pretty_model, quant)] = rows
            if quant not in quant_labels:
                quant_labels.append(quant)
        del fp_reader

    if not series:
        console.print("[yellow]No layer-wise errors to plot[/yellow]")
        return None

    quant_labels = sorted(quant_labels, key=_quant_rank)
    model_names = list(dict.fromkeys(model_names))
    n_models, n_quants = len(model_names), len(quant_labels)
    fig, axes = plt.subplots(
        n_models,
        n_quants,
        figsize=(max(3.2 * n_quants, 8), max(3.0 * n_models, 4.5)),
        squeeze=False,
        sharex=True,
        sharey=False,
    )
    csv_rows = []

    for i, model in enumerate(model_names):
        for j, quant in enumerate(quant_labels):
            ax = axes[i, j]
            rows = series.get((model, quant))
            if not rows:
                ax.axis("off")
                continue
            by_kind: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
            for row in rows:
                by_kind[row["tensor"]].append((row["layer"], row["rel_l2"]))
                csv_rows.append(
                    {
                        "model": model,
                        "quant": quant,
                        "layer": row["layer"],
                        "tensor": row["tensor"],
                        "rel_l2": row["rel_l2"],
                    }
                )
            layers_all = sorted({layer for points in by_kind.values() for layer, _ in points})
            mean_y = []
            for layer in layers_all:
                vals = [err for points in by_kind.values() for lyr, err in points if lyr == layer]
                mean_y.append(float(np.mean(vals)) if vals else float("nan"))
            for _, label, _, _ in _KIND_SPECS:
                points = by_kind.get(label)
                if not points:
                    continue
                points = sorted(points)
                color, ls = _KIND_STYLE[label]
                ax.plot(
                    [p[0] for p in points],
                    [p[1] for p in points],
                    color=color,
                    ls=ls,
                    lw=1.15,
                    label=label,
                    zorder=2,
                )
            ax.plot(
                layers_all,
                mean_y,
                color="#222222",
                ls=":",
                lw=1.6,
                label="mean",
                zorder=3,
            )
            if i == 0:
                ax.set_title(quant, fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{model}\nRelative L2 error", fontsize=9)
            if i == n_models - 1:
                ax.set_xlabel("Layer", fontsize=9)
            mean_err = float(np.nanmean(mean_y)) if mean_y else float("nan")
            ax.text(
                0.97,
                0.94,
                f"mean={mean_err:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#444444",
            )
            _style_axes(ax)
            if i == 0 and j == 0:
                ax.legend(
                    loc="upper left",
                    fontsize=6.5,
                    ncol=2,
                    frameon=True,
                    borderpad=0.3,
                    labelspacing=0.25,
                    handlelength=1.6,
                )

    fig.suptitle("Layer-wise quantization error vs FP16", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _finish_plot(pd.DataFrame(csv_rows), output_path, "layerwise_quant_error.png")


_CMP_FIXED = (
    ("perplexity", "perplexity", "ppl"),
    ("perplexity_long", "perplexity long", "ppl"),
    ("delta_ppl", "delta ppl", "frac"),
    ("kl_divergence", "KL", "kl"),
    ("top1_match", "top-1 match", "frac"),
    ("size_gb", "file size", "size"),
    ("bits_per_weight", "bits/weight", "bpw"),
    ("compression_ratio", "compression ratio", "ratio"),
    ("size_reduction", "size reduction", "frac"),
    ("latency_ms", "ms/tok", "ms"),
    ("decode_tok_s", "decode tok/s", "toks"),
    ("prefill_tok_s", "prefill tok/s", "toks"),
    ("speedup", "speedup", "ratio"),
    ("prefill_speedup", "prefill speedup", "ratio"),
    ("vram_gb", "VRAM", "vram"),
)


def _cmp_format(kind: str, value) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    x = float(value)
    if kind == "ppl":
        return f"{x:.4f}"
    if kind == "kl":
        return f"{x:.4f}"
    if kind == "frac":
        return f"{x:.4f}"
    if kind == "size":
        return f"{x:.1f}G" if x >= 10 else f"{x:.2f}G"
    if kind == "bpw":
        return f"{x:.1f}"
    if kind == "ms":
        return f"{x:.0f}" if abs(x - round(x)) < 0.05 else f"{x:.1f}"
    if kind == "toks":
        return f"{x:.1f}"
    if kind == "ratio":
        return f"{x:.3f}"
    if kind == "vram":
        return f"{x:.2f}"
    return f"{x:.4f}"


def _cmp_task_specs(rows: List[dict]) -> List[Tuple[str, str, str]]:
    tasks: List[str] = []
    for row in rows:
        for key in row:
            if key.startswith("acc_") and not key.startswith("acc_retention_"):
                task = key[len("acc_") :]
                if task not in tasks:
                    tasks.append(task)
    ordered = [t for t in _TASK_ORDER if t in tasks]
    ordered += [t for t in tasks if t not in ordered]
    specs: List[Tuple[str, str, str]] = []
    for task in ordered:
        specs.extend(
            (
                (f"acc_{task}", f"acc {task}", "frac"),
                (f"delta_acc_{task}", f"delta acc {task}", "frac"),
                (f"acc_retention_{task}", f"acc retention {task}", "frac"),
                (f"efficiency_{task}", f"efficiency {task}", "frac"),
            )
        )
    return specs


def plot_comparison_table(
    results: List[EvaluationResult],
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """Wide master table: models as row groups, measures as rows, quants as columns."""
    console.print("[cyan]Creating comparison master table...[/cyan]")
    if not results:
        console.print("[yellow]No evaluation results for comparison table[/yellow]")
        return None

    summaries = [summary_row(r) for r in results]
    cells: Dict[Tuple[str, str], dict] = {}
    for row in summaries:
        model = _pretty_model(row.get("model") or "")
        quant = _pretty_quant(row.get("quant") or "")
        cells[(model, quant)] = row

    models = list(_series_for([_pretty_model(r.model_name) for r in results]).keys())
    quants = _ordered_quants(_pretty_quant(r.quantization_type) for r in results)
    measure_specs = list(_CMP_FIXED) + _cmp_task_specs(summaries)

    wide_rows = []
    cell_text = []
    for model in models:
        first = True
        for field, label, kind in measure_specs:
            values = []
            any_val = False
            for quant in quants:
                raw = cells.get((model, quant), {}).get(field)
                if raw is not None:
                    any_val = True
                values.append(_cmp_format(kind, raw))
            if not any_val:
                continue
            display_model = model if first else ""
            first = False
            wide_rows.append({"model": display_model, "measure": label, **dict(zip(quants, values))})
            cell_text.append([display_model, label, *values])

    if not wide_rows:
        console.print("[yellow]No comparison table rows to plot[/yellow]")
        return None

    df = pd.DataFrame(wide_rows)
    headers = ["Model", "Measure", *quants]
    n_rows, n_cols = len(cell_text), len(headers)
    fig, ax = plt.subplots(figsize=(max(1.35 * n_cols, 11), max(0.38 * n_rows + 1.1, 4)))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    header_color = "#d9d9d9"
    zebra = "#f3f3f3"
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#b0b0b0")
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor(zebra)
        if col <= 1:
            cell._loc = "left"
            cell.PAD = 0.04
    ax.set_title("Comparison", fontsize=14, fontweight="bold", pad=12)
    return _finish_plot(df, output_path, "comparison.png")


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
        ("comparison", plot_comparison_table, "comparison.png"),
        ("perplexity_vs_size", plot_size_vs_quality, "perplexity_vs_size.png"),
        ("perplexity_vs_throughput", plot_perplexity_vs_throughput, "perplexity_vs_throughput.png"),
        ("delta_ppl_vs_bits", plot_delta_ppl_vs_bits, "delta_ppl_vs_bits.png"),
        ("perplexity_drop", plot_perplexity_drop, "perplexity_drop.png"),
        ("kl_top1_vs_bits", plot_kl_top1_vs_bits, "kl_top1_vs_bits.png"),
        ("prefill_vs_decode", plot_prefill_vs_decode, "prefill_vs_decode.png"),
        ("perplexity_short_vs_long", plot_perplexity_short_vs_long, "perplexity_short_vs_long.png"),
        ("efficiency_pareto", plot_efficiency_pareto, "efficiency_pareto.png"),
        ("accuracy_vs_size", plot_accuracy_vs_size, "accuracy_vs_size.png"),
        ("accuracy_drop", plot_accuracy_drop, "accuracy_drop.png"),
        ("accuracy_vs_throughput", plot_accuracy_vs_throughput, "accuracy_vs_throughput.png"),
        ("weight_distribution", plot_weight_distributions, "weight_distribution.png"),
        ("layerwise_quant_error", plot_layerwise_quant_error, "layerwise_quant_error.png"),
    ]
    chart_paths = {}
    for name, fn, filename in jobs:
        path = fn(results, chart_dir / filename)
        if path:
            chart_paths[name] = path

    console.print(f"\n[bold green]Generated {len(chart_paths)} chart(s)[/bold green]")
    return chart_paths

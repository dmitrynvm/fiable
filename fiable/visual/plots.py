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
    QUANT_TYPES,
    parse_quant_spec,
    llama_src_dir,
)
from fiable.core.evaluate import EvaluationResult
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


# High-bit → low-bit GGUF methods (baselines first, then QUANT_TYPES).
_QUANT_ORDER = (
    "FP32",
    "F32",
    "FP16",
    "F16",
    "BF16",
    "Q8_0",
    "Q6_K",
    "Q5_K_M",
    "Q5_K_S",
    "Q5_0",
    "Q4_K_M",
    "Q4_K_S",
    "Q4_0",
    "Q3_K_L",
    "Q3_K_M",
    "Q3_K_S",
    "Q2_K",
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
    for name in QUANT_TYPES:
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
    colors = _model_colors(unique_models)
    # Low compression (FP16, ratio ~1) → high compression (Q2_K).
    quants = _ordered_quants(df["quant"])
    width = 0.8 / max(len(unique_models), 1)
    x = range(len(quants))
    for i, model in enumerate(unique_models):
        model_df = df[df["model"] == model].set_index("quant")
        heights = [float(model_df.loc[q, "ratio"]) if q in model_df.index else 0 for q in quants]
        offset = width * (i - (len(unique_models) - 1) / 2)
        bars = ax.bar(
            [j + offset for j in x],
            heights,
            width,
            label=model,
            color=colors[model],
            edgecolor=_BAR_EDGE,
            linewidth=0.7,
        )
        for bar, quant in zip(bars, quants):
            if quant in model_df.index and pd.notna(model_df.loc[quant, "bpw"]):
                ax.annotate(
                    f"{model_df.loc[quant, 'bpw']:.1f} bpw",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                    color=colors[model],
                )

    ax.axhline(1.0, color="#b3b3b3", linestyle="--", linewidth=1.1, alpha=0.9, zorder=0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(quants, rotation=30, ha="right")
    ax.set_xlabel("Quantization method", fontsize=12, fontweight="bold")
    ax.set_ylabel("Ratio (×)", fontsize=12, fontweight="bold")
    ax.set_title("Compression", fontsize=14, fontweight="bold")
    ax.legend(loc="best", frameon=True)
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


_WEIGHT_SAMPLE_N = 400_000
_WEIGHT_BINS = 96
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
        sharey="row",
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
        for j, quant in enumerate(quant_labels):
            ax = axes[i, j]
            arr = samples.get((model, quant))
            if arr is None:
                ax.axis("off")
                continue
            density, edges = np.histogram(arr, bins=bins, density=True)
            ax.hist(
                arr,
                bins=bins,
                density=True,
                color=colors[j],
                edgecolor="none",
                alpha=0.88,
                zorder=2,
            )
            if not is_baseline_quant(quant):
                ax.plot(
                    ref_centers,
                    ref_density,
                    color="#444444",
                    lw=1.05,
                    ls="--",
                    alpha=0.8,
                    zorder=3,
                    label="FP16",
                )
            ax.set_xlim(*xlim)
            if i == 0:
                ax.set_title(quant, fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{model}\nDensity", fontsize=9)
            if i == n_models - 1:
                ax.set_xlabel("Weight value", fontsize=9)
            ax.text(
                0.97,
                0.94,
                f"σ={float(arr.std()):.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#444444",
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
        ("compression", plot_compression_ratio, "compression.png"),
        ("perplexity_vs_throughput", plot_perplexity_vs_throughput, "perplexity_vs_throughput.png"),
        ("accuracy", plot_accuracy_vs_size, "accuracy.png"),
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

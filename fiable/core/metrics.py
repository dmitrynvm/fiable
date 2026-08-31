"""Shared compression metrics: identity, GGUF metadata, NVML, relative deltas."""

from __future__ import annotations

import struct
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from fiable.config.settings import STORE_DIR, OUTPUT_DIR, QUANT_TYPES, MODELS, get_fp16_path


BASELINE_QUANTS = ("FP16", "F16", "BF16", "FP32", "F32")
QUANT_ALIASES = {
    "F16": "FP16",
    "FP16": "FP16",
    "BF16": "BF16",
    "F32": "FP32",
    "FP32": "FP32",
}

_GGUF_STRING = 8
_GGUF_ARRAY = 9
_GGUF_UINT64 = 10
_GGUF_INT64 = 11
_GGUF_SIZES = {
    0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
    10: 8, 11: 8, 12: 8,
}


def known_quant_suffixes() -> List[str]:
    suffixes = list(QUANT_TYPES) + list(QUANT_ALIASES.keys())
    return sorted(set(suffixes), key=len, reverse=True)


def parse_model_identity(path: Path) -> Tuple[str, str]:
    """Return (model_name, quantization_type) from a GGUF filename."""
    stem = path.stem
    upper = stem.upper()
    for quant in known_quant_suffixes():
        token = quant.upper()
        for sep in ("-", "_"):
            suffix = sep + token
            if upper.endswith(suffix):
                name = stem[: -len(suffix)]
                return name, QUANT_ALIASES.get(token, quant)
    if "-" in stem:
        return "-".join(stem.split("-")[:-1]), stem.split("-")[-1]
    return stem, "unknown"


def is_baseline_quant(quant: str) -> bool:
    return quant.upper() in {q.upper() for q in BASELINE_QUANTS} | {"FP16", "BF16", "FP32"}


def default_eval_paths() -> List[Path]:
    """FP16 baselines plus quantized GGUFs in store/."""
    seen = set()
    paths: List[Path] = []
    for model in MODELS:
        fp16 = get_fp16_path(model)
        if fp16.exists() and fp16.resolve() not in seen:
            paths.append(fp16)
            seen.add(fp16.resolve())
    for pattern in ("*fp16.gguf", "*FP16.gguf", "*f16.gguf"):
        for path in sorted(STORE_DIR.glob(pattern)):
            if path.resolve() not in seen:
                paths.append(path)
                seen.add(path.resolve())
    if OUTPUT_DIR.exists():
        for path in sorted(OUTPUT_DIR.glob("*.gguf")):
            if path.resolve() not in seen:
                paths.append(path)
                seen.add(path.resolve())
    return paths


def read_gguf_scalars(path: Path) -> Dict[str, object]:
    """Read non-array GGUF metadata keys (skips tokenizer tables)."""
    meta: Dict[str, object] = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return meta
        version = struct.unpack("<I", f.read(4))[0]
        if version < 2:
            return meta
        f.read(8)  # tensor_count
        n_kv = struct.unpack("<Q", f.read(8))[0]
        for _ in range(min(int(n_kv), 1024)):
            key = _read_gguf_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            if vtype == _GGUF_ARRAY:
                _skip_gguf_value(f, vtype)
                continue
            meta[key] = _read_gguf_value(f, vtype)
    return meta


def _estimate_params_from_arch(meta: Dict[str, object]) -> Optional[int]:
    arch = meta.get("general.architecture")
    if not isinstance(arch, str):
        return None

    def _int(key: str, default: Optional[int] = None) -> Optional[int]:
        val = meta.get(key, default)
        if isinstance(val, bool) or val is None:
            return default
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    n_layer = _int(f"{arch}.block_count")
    d_model = _int(f"{arch}.embedding_length")
    d_ffn = _int(f"{arch}.feed_forward_length")
    n_head = _int(f"{arch}.attention.head_count")
    n_kv = _int(f"{arch}.attention.head_count_kv", n_head)
    vocab = _int(f"{arch}.vocab_size")
    if not all([n_layer, d_model, d_ffn, n_head, n_kv, vocab]):
        return None
    head_dim = _int(f"{arch}.attention.key_length") or (d_model // n_head)
    val_dim = _int(f"{arch}.attention.value_length") or head_dim
    q = d_model * n_head * head_dim
    k = d_model * n_kv * head_dim
    v = d_model * n_kv * val_dim
    o = n_head * head_dim * d_model
    ffn = 3 * d_model * d_ffn
    norms = 2 * d_model
    embed = vocab * d_model
    lm_head = vocab * d_model
    return embed + n_layer * (q + k + v + o + ffn + norms) + d_model + lm_head


def read_gguf_parameter_count(path: Path) -> Optional[int]:
    """Parameter count from GGUF metadata, or an architecture-based estimate."""
    try:
        meta = read_gguf_scalars(path)
    except Exception:
        return None
    raw = meta.get("general.parameter_count")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return int(raw)
    return _estimate_params_from_arch(meta)


def _read_gguf_string(f) -> str:
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", errors="replace")


_GGUF_FMT = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
    10: "Q", 11: "q", 12: "d",
}


def _read_gguf_value(f, vtype: int):
    if vtype == _GGUF_STRING:
        return _read_gguf_string(f)
    if vtype == 7:
        return bool(f.read(1)[0])
    if vtype == _GGUF_ARRAY:
        etype = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        return [_read_gguf_value(f, etype) for _ in range(n)]
    fmt = _GGUF_FMT.get(vtype)
    if fmt is None:
        raise ValueError(f"unsupported GGUF type {vtype}")
    return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]


def _skip_gguf_value(f, vtype: int) -> None:
    if vtype == _GGUF_STRING:
        n = struct.unpack("<Q", f.read(8))[0]
        f.seek(n, 1)
        return
    if vtype == _GGUF_ARRAY:
        etype = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            _skip_gguf_value(f, etype)
        return
    size = _GGUF_SIZES.get(vtype)
    if size is None:
        raise ValueError(f"unsupported GGUF type {vtype}")
    f.seek(size, 1)


def bits_per_weight(file_size_gb: float, n_params: Optional[int]) -> Optional[float]:
    if not n_params or n_params <= 0 or file_size_gb <= 0:
        return None
    return file_size_gb * (1024 ** 3) * 8.0 / n_params


def _harmonic_mean(a: float, b: float) -> Optional[float]:
    if a <= 0 or b <= 0:
        return None
    return 2.0 * a * b / (a + b)


def pick_baseline(group: Iterable) -> Optional[object]:
    """Prefer FP16/BF16, then Q8_0, then the largest file in the group."""
    items = list(group)
    if not items:
        return None
    for preferred in ("FP16", "BF16", "F16", "FP32", "F32"):
        for item in items:
            if str(getattr(item, "quantization_type", "")).upper() == preferred:
                return item
    for item in items:
        if str(getattr(item, "quantization_type", "")).upper() == "Q8_0":
            return item
    return max(items, key=lambda r: getattr(r, "file_size_gb", 0.0) or 0.0)


def annotate_relative_metrics(results: list) -> None:
    """Fill compression_ratio, bits_per_weight, delta_ppl, delta_acc, acc_retention, speedup, efficiency vs baseline."""
    by_model: Dict[str, list] = defaultdict(list)
    for result in results:
        by_model[result.model_name].append(result)

    for group in by_model.values():
        baseline = pick_baseline(group)
        if baseline is None:
            continue
        base_size = baseline.file_size_gb or 0.0
        base_ppl = baseline.perplexity
        base_acc = dict(baseline.benchmarks or {})
        for result in group:
            result.baseline_quant = baseline.quantization_type
            if base_size > 0 and result.file_size_gb:
                result.compression_ratio = base_size / result.file_size_gb
                result.size_reduction = max(0.0, 1.0 - float(result.file_size_gb) / base_size)
            n_params = result.n_params or getattr(baseline, "n_params", None)
            if n_params and not result.n_params:
                result.n_params = n_params
            result.bits_per_weight = bits_per_weight(result.file_size_gb, result.n_params)
            if base_ppl and result.perplexity:
                result.delta_ppl = result.perplexity / base_ppl - 1.0
            deltas: Dict[str, float] = {}
            retention: Dict[str, float] = {}
            efficiency: Dict[str, float] = {}
            size_red = result.size_reduction
            for task, acc in (result.benchmarks or {}).items():
                if task not in base_acc:
                    continue
                orig = float(base_acc[task])
                comp = float(acc)
                deltas[task] = orig - comp
                if orig:
                    retention[task] = comp / orig
                    score = _harmonic_mean(retention[task], size_red or 0.0)
                    if score is not None:
                        efficiency[task] = score
            result.delta_acc = deltas
            result.acc_retention = retention
            result.efficiency_score = efficiency
            base_lat = getattr(baseline, "throughput_latency_ms", None)
            comp_lat = result.throughput_latency_ms
            if base_lat and comp_lat:
                result.speedup = float(base_lat) / float(comp_lat)
            else:
                base_tps = getattr(baseline, "throughput_tokens_per_sec", None)
                comp_tps = result.throughput_tokens_per_sec
                if base_tps and comp_tps:
                    result.speedup = float(comp_tps) / float(base_tps)
            base_ttft = getattr(baseline, "ttft_ms", None)
            if base_ttft and result.ttft_ms:
                result.prefill_speedup = float(base_ttft) / float(result.ttft_ms)


def _nvml_used_bytes() -> Optional[int]:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
    except Exception:
        return None


def _smi_used_bytes() -> Optional[int]:
    import subprocess

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
        mb = max(float(line.strip()) for line in out.splitlines() if line.strip())
        return int(mb * 1024 * 1024)
    except Exception:
        return None


@contextmanager
def peak_vram_sampler(interval: float = 0.05):
    """Yield peak VRAM used by the wrapped block (device peak minus start)."""
    state = {"peak_vram_gb": None}
    stop = threading.Event()
    baseline = _nvml_used_bytes() or _smi_used_bytes() or 0
    peak = {"bytes": baseline}

    def _poll():
        while not stop.is_set():
            used = _nvml_used_bytes()
            if used is None:
                used = _smi_used_bytes()
            if used:
                peak["bytes"] = max(peak["bytes"], used)
            time.sleep(interval)

    thread = threading.Thread(target=_poll, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        stop.set()
        thread.join(timeout=2)
        used = _nvml_used_bytes() or _smi_used_bytes()
        if used:
            peak["bytes"] = max(peak["bytes"], used)
        delta = max(0, peak["bytes"] - baseline)
        if delta or peak["bytes"]:
            state["peak_vram_gb"] = delta / (1024 ** 3)


def pareto_mask(xs: List[float], ys: List[float], minimize_x: bool = True, minimize_y: bool = True) -> List[bool]:
    """True for non-dominated points (Pareto front)."""
    n = len(xs)
    mask = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x_better = (xs[j] <= xs[i]) if minimize_x else (xs[j] >= xs[i])
            y_better = (ys[j] <= ys[i]) if minimize_y else (ys[j] >= ys[i])
            x_strict = (xs[j] < xs[i]) if minimize_x else (xs[j] > xs[i])
            y_strict = (ys[j] < ys[i]) if minimize_y else (ys[j] > ys[i])
            if x_better and y_better and (x_strict or y_strict):
                mask[i] = False
                break
    return mask

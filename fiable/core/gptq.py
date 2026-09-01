"""GPTQ (GPTQModel) quantization, then dequant GGUF for llama.cpp eval."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from rich.console import Console

from fiable.config.settings import (
    DATASETS_DIR,
    ModelConfig,
    STORE_DIR,
    ensure_llama_src,
    get_gptq_packed_dir,
    get_quant_meta_path,
    get_quantized_path,
)
from fiable.utils import helpers

if TYPE_CHECKING:
    from fiable.core.quantize import QuantizationResult

console = Console()

GPTQ_BITS = 4
GPTQ_GROUP_SIZE = 128
GPTQ_DESC_ACT = True
CALIB_SAMPLES = 128
CALIB_SEQLEN = 2048
CALIB_MIN_CHARS = 10


def dir_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return helpers.get_file_size_gb(path)
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 ** 3)


def write_quant_meta(
    gguf_path: Path,
    *,
    quantization_type: str,
    packed_dir: Path,
    bits: int = GPTQ_BITS,
    group_size: int = GPTQ_GROUP_SIZE,
    desc_act: bool = GPTQ_DESC_ACT,
) -> Path:
    meta = {
        "quantization_type": quantization_type,
        "method": "gptq",
        "bits": bits,
        "group_size": group_size,
        "desc_act": desc_act,
        "packed_dir": str(packed_dir),
        "packed_size_gb": round(dir_size_gb(packed_dir), 6),
        "gguf_path": str(gguf_path),
        "skip_throughput": True,
        "note": (
            "GGUF is F16 of GPTQ-reconstructed weights for llama.cpp PPL/KL; "
            "size_gb uses packed_size_gb, not the GGUF file."
        ),
    }
    meta_path = gguf_path.with_suffix(".meta.json")
    helpers.save_json(meta, meta_path)
    return meta_path


def load_quant_meta(gguf_path: Path) -> Optional[dict]:
    meta_path = Path(gguf_path).with_suffix(".meta.json")
    if not meta_path.is_file():
        return None
    try:
        return helpers.load_json(meta_path)
    except (OSError, json.JSONDecodeError):
        return None


def effective_size_gb(gguf_path: Path) -> float:
    """Packed GPTQ size when a sidecar exists; otherwise the GGUF file size."""
    meta = load_quant_meta(gguf_path)
    if meta:
        packed = meta.get("packed_size_gb")
        if packed:
            return float(packed)
        packed_dir = meta.get("packed_dir")
        if packed_dir:
            size = dir_size_gb(Path(packed_dir))
            if size > 0:
                return size
    return helpers.get_file_size_gb(Path(gguf_path))


def _ensure_torch() -> None:
    try:
        import torch  # noqa: F401

        return
    except ImportError:
        pass
    console.print("[dim]Installing torch (required by GPTQModel)...[/dim]")
    helpers.run_command(
        [sys.executable, "-m", "pip", "install", "-q", "torch"],
        shell=False,
    )


def _ensure_gptqmodel() -> None:
    _ensure_torch()
    try:
        import gptqmodel  # noqa: F401

        return
    except ImportError:
        pass
    console.print("[dim]Installing gptqmodel (pip install fiable[gptq])...[/dim]")
    helpers.run_command(
        [sys.executable, "-m", "pip", "install", "-q", "gptqmodel"],
        shell=False,
    )


def _wikitext2_train_texts(n_take: int = 2048) -> List[str]:
    """Non-empty WikiText-2 train paragraphs from the local ggml zip (not Hub `wikitext`)."""
    from fiable.core.evaluate import _ensure_wikitext

    train_path = _ensure_wikitext("train")
    texts: List[str] = []
    with open(train_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            text = line.strip()
            if len(text) < CALIB_MIN_CHARS:
                continue
            texts.append(text)
            if len(texts) >= n_take:
                break
    if not texts:
        raise RuntimeError("WikiText-2 train split produced no calibration strings")
    return texts


def _convert_packed_to_gguf(packed_dir: Path, gguf_path: Path) -> None:
    """Dequantize GPTQ with GPTQModel, then convert dense HF weights to F16 GGUF.

    llama.cpp's convert path unpacks GPTQ itself and is wrong for some
    Llama + desc_act checkpoints (WikiText PPL explodes while KL/GSM8K look fine).
    """
    from fiable.core.quantize import _ensure_hf_convert_deps
    import torch
    from gptqmodel.utils.model_dequant import dequantize_model

    llama_cpp_dir = ensure_llama_src()
    _ensure_hf_convert_deps()
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.is_file():
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found: {convert_script}")

    dequant_dir = packed_dir.parent / f"{packed_dir.name}-dequant-f16"
    if dequant_dir.exists():
        shutil.rmtree(dequant_dir)
    console.print("[cyan]Dequantizing GPTQ → dense FP16 (GPTQModel), then GGUF...[/cyan]")
    dequantize_model(
        packed_dir,
        dequant_dir,
        target_dtype=torch.float16,
        device=None,
    )
    try:
        cmd = [
            sys.executable,
            str(convert_script),
            str(dequant_dir),
            "--outfile",
            str(gguf_path),
            "--outtype",
            "f16",
            "--no-lazy",
        ]
        helpers.run_command(cmd, shell=False, capture_output=False)
    finally:
        shutil.rmtree(dequant_dir, ignore_errors=True)


def quantize_gptq(
    model: ModelConfig,
    quant_type: str,
    force: bool = False,
) -> "QuantizationResult":
    """GPTQ-quantize HF weights, save packed dir, write dequant F16 GGUF + sidecar."""
    from fiable.core.quantize import QuantizationResult

    hf_dir = STORE_DIR / model.local_dir
    packed_dir = get_gptq_packed_dir(model, quant_type)
    gguf_path = get_quantized_path(model, quant_type)
    meta_path = get_quant_meta_path(model, quant_type)

    if not hf_dir.is_dir():
        msg = f"Model directory not found: {hf_dir}. Run download first."
        console.print(f"[red]{msg}[/red]")
        return QuantizationResult(
            model_name=model.name,
            quantization_type=quant_type,
            success=False,
            error=msg,
        )

    packed_ready = packed_dir.is_dir() and any(packed_dir.glob("*.safetensors"))
    if (
        packed_ready
        and gguf_path.exists()
        and meta_path.exists()
        and not force
    ):
        file_size = effective_size_gb(gguf_path)
        console.print(
            f"[yellow]Skipping {quant_type} - already exists "
            f"(packed {file_size:.2f} GB)[/yellow]"
        )
        return QuantizationResult(
            model_name=model.name,
            quantization_type=quant_type,
            success=True,
            output_path=gguf_path,
            file_size_gb=file_size,
            already_exists=True,
        )

    start_time = time.time()
    try:
        _ensure_gptqmodel()
        need_quant = force or not packed_ready
        if need_quant:
            from gptqmodel import GPTQModel, QuantizeConfig

            console.print(
                f"[cyan]GPTQ {quant_type}: {GPTQ_BITS}-bit g{GPTQ_GROUP_SIZE} "
                f"act-order, WikiText-2 train calib ({CALIB_SAMPLES}×{CALIB_SEQLEN})...[/cyan]"
            )
            calib = _wikitext2_train_texts(n_take=max(512, CALIB_SAMPLES * 4))
            qcfg = QuantizeConfig(
                bits=GPTQ_BITS,
                group_size=GPTQ_GROUP_SIZE,
                desc_act=GPTQ_DESC_ACT,
            )
            gptq_model = GPTQModel.load(str(hf_dir), qcfg)
            quantize_kwargs = {
                "batch_size": 1,
                "calibration_dataset_concat_size": CALIB_SEQLEN,
                "calibration_dataset_min_length": CALIB_MIN_CHARS,
            }
            try:
                gptq_model.quantize(calib[: CALIB_SAMPLES * 8], **quantize_kwargs)
            except TypeError:
                gptq_model.quantize(calib[: CALIB_SAMPLES * 8], batch_size=1)

            packed_dir.mkdir(parents=True, exist_ok=True)
            gptq_model.save(str(packed_dir))
            del gptq_model
            try:
                import gc
                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        else:
            console.print(
                f"[cyan]Reusing packed GPTQ at {packed_dir.name}, rebuilding GGUF...[/cyan]"
            )

        console.print(f"[cyan]Converting GPTQ checkpoint to GGUF F16 ({gguf_path.name})...[/cyan]")
        _convert_packed_to_gguf(packed_dir, gguf_path)
        write_quant_meta(
            gguf_path,
            quantization_type=quant_type,
            packed_dir=packed_dir,
        )
        duration = time.time() - start_time
        file_size = effective_size_gb(gguf_path)
        console.print(
            f"[green]✓ Created {quant_type} (packed {file_size:.2f} GB) "
            f"in {helpers.format_duration(duration)}[/green]"
        )
        return QuantizationResult(
            model_name=model.name,
            quantization_type=quant_type,
            success=True,
            output_path=gguf_path,
            file_size_gb=file_size,
            duration_seconds=duration,
        )
    except Exception as e:
        duration = time.time() - start_time
        console.print(f"[red]✗ GPTQ {quant_type} failed: {e}[/red]")
        return QuantizationResult(
            model_name=model.name,
            quantization_type=quant_type,
            success=False,
            error=str(e),
            duration_seconds=duration,
        )

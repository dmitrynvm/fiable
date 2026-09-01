"""EvoPress mixed-precision search over uniform llama.cpp GGUF types."""

from __future__ import annotations

import os
import random
import re
import shlex
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from rich.console import Console

from fiable.config.settings import (
    DATASETS_DIR,
    ModelConfig,
    get_fp16_path,
    get_quant_meta_path,
    get_quantized_path,
    llama_binary,
    llama_src_dir,
)
from fiable.utils import helpers

if TYPE_CHECKING:
    from fiable.core.quantize import QuantizationResult

console = Console()

LEVELS: Tuple[str, ...] = ("Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K")
LEVEL_BITS = {"Q2_K": 2, "Q3_K_M": 3, "Q4_K_M": 4, "Q5_K_M": 5, "Q6_K": 6}
NONBLOCK_SOURCE = "Q4_K_M"
_BLK_RE = re.compile(r"^blk\.(\d+)\.")
_SKIP_KV = ("GGUF.", "split.")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    return float(raw)


def _env_int_list(name: str, default: Sequence[int]) -> List[int]:
    raw = os.environ.get(name)
    if not raw:
        return list(default)
    return [int(x.strip()) for x in raw.replace(" ", ",").split(",") if x.strip()]


def _import_gguf():
    try:
        import gguf
        from gguf.constants import GGUFValueType

        return gguf, GGUFValueType
    except ImportError:
        src = llama_src_dir() / "gguf-py"
        if src.is_dir() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
        import gguf
        from gguf.constants import GGUFValueType

        return gguf, GGUFValueType


def mean_bits(block_types: Sequence[str]) -> float:
    if not block_types:
        return 0.0
    return sum(LEVEL_BITS[t] for t in block_types) / len(block_types)


def is_feasible(
    block_types: Sequence[str],
    target: float = 4.0,
    tol: float = 0.05,
) -> bool:
    return abs(mean_bits(block_types) - target) <= tol + 1e-9


def n_blocks_from_tensors(names: Sequence[str]) -> int:
    idxs = []
    for name in names:
        match = _BLK_RE.match(name)
        if match:
            idxs.append(int(match.group(1)))
    if not idxs:
        raise ValueError("No blk.N tensors found in GGUF")
    return max(idxs) + 1


def random_config(n_layers: int, rng: random.Random, target: float, tol: float) -> List[str]:
    for _ in range(20_000):
        cfg = [rng.choice(LEVELS) for _ in range(n_layers)]
        if is_feasible(cfg, target, tol):
            return cfg
    # Deterministic mix of 2-bit and 6-bit around the target.
    cfg = [NONBLOCK_SOURCE] * n_layers
    lo, hi = 0, n_layers - 1
    while not is_feasible(cfg, target, tol) and lo < hi:
        if mean_bits(cfg) > target:
            cfg[lo] = "Q2_K"
            lo += 1
        else:
            cfg[hi] = "Q6_K"
            hi -= 1
    return cfg


def mutate_config(
    parent: Sequence[str],
    rng: random.Random,
    target: float,
    tol: float,
) -> Optional[List[str]]:
    n = len(parent)
    k = min(rng.randint(1, 3), rng.randint(1, 3), max(1, n // 2))
    for _ in range(80):
        child = list(parent)
        can_up = [i for i, t in enumerate(child) if LEVELS.index(t) < len(LEVELS) - 1]
        can_down = [i for i, t in enumerate(child) if LEVELS.index(t) > 0]
        if len(can_up) < k or len(can_down) < k:
            k = min(k, len(can_up), len(can_down))
        if k < 1:
            return None
        up = rng.sample(can_up, k)
        down_pool = [i for i in can_down if i not in set(up)]
        if len(down_pool) < k:
            continue
        down = rng.sample(down_pool, k)
        for i in up:
            child[i] = LEVELS[LEVELS.index(child[i]) + 1]
        for i in down:
            child[i] = LEVELS[LEVELS.index(child[i]) - 1]
        if is_feasible(child, target, tol) and child != list(parent):
            return child
    return None


def _tensor_map(reader) -> Dict[str, object]:
    return {t.name: t for t in reader.tensors}


def stitch_gguf(
    block_types: Sequence[str],
    sources: Dict[str, Path],
    outfile: Path,
) -> None:
    """Copy blk.i tensors from that block's GGUF; non-block tensors from Q4_K_M."""
    gguf, GGUFValueType = _import_gguf()
    template_path = sources[NONBLOCK_SOURCE]
    template = gguf.GGUFReader(str(template_path))
    readers = {NONBLOCK_SOURCE: template}
    maps = {NONBLOCK_SOURCE: _tensor_map(template)}
    needed = set(block_types) | {NONBLOCK_SOURCE}
    for level in needed:
        if level in readers:
            continue
        path = sources.get(level)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"EvoPress database GGUF missing for {level}")
        readers[level] = gguf.GGUFReader(str(path))
        maps[level] = _tensor_map(readers[level])

    arch_field = template.get_field("general.architecture")
    arch = arch_field.contents() if arch_field else "llama"
    writer = gguf.GGUFWriter(str(outfile), arch, endianess=template.endianess)
    writer.data_alignment = getattr(template, "alignment", writer.data_alignment)

    for field in template.fields.values():
        if field.name == "general.architecture" or field.name.startswith(_SKIP_KV):
            continue
        if not field.types:
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None
        value = field.contents()
        if value is None:
            continue
        writer.add_key_value(field.name, value, val_type, sub_type=sub_type)

    selected = []
    for tensor in template.tensors:
        match = _BLK_RE.match(tensor.name)
        if match:
            idx = int(match.group(1))
            if idx >= len(block_types):
                raise IndexError(f"block {idx} not in config of length {len(block_types)}")
            src = maps[block_types[idx]].get(tensor.name)
            if src is None:
                raise KeyError(f"{tensor.name} missing in {block_types[idx]} GGUF")
            selected.append(src)
        else:
            selected.append(tensor)

    for tensor in selected:
        writer.add_tensor_info(
            tensor.name,
            tensor.data.shape,
            tensor.data.dtype,
            tensor.data.nbytes,
            tensor.tensor_type,
        )

    outfile.parent.mkdir(parents=True, exist_ok=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in selected:
        writer.write_tensor_data(
            np.ascontiguousarray(tensor.data),
            tensor_endianess=template.endianess,
        )
    writer.close()


def _dump_logits(fp16_path: Path, dataset_path: Path, logits_path: Path, chunks: int) -> None:
    from fiable.core.evaluate import _run_stream_with_ppl_bar

    ppl_bin = llama_binary("llama-perplexity")
    dump = _run_stream_with_ppl_bar(
        f"{shlex.quote(str(ppl_bin))} "
        f"-m {shlex.quote(str(fp16_path))} "
        f"-f {shlex.quote(str(dataset_path))} "
        f"-c 512 -ngl 99 --chunks {int(chunks)} "
        f"--save-all-logits {shlex.quote(str(logits_path))}",
        "EvoPress KL baseline logits",
        total=int(chunks),
    )
    if dump.returncode != 0 or not logits_path.exists():
        err = (dump.stderr or dump.stdout or "failed to dump logits").strip()[:400]
        raise RuntimeError(f"EvoPress KL baseline dump failed: {err}")


def _kl_vs_logits(model_path: Path, dataset_path: Path, logits_path: Path, chunks: int) -> float:
    from fiable.core.evaluate import _parse_kl_output

    ppl_bin = llama_binary("llama-perplexity")
    result = helpers.run_command(
        [
            str(ppl_bin),
            "-m",
            str(model_path),
            "-f",
            str(dataset_path),
            "-c",
            "512",
            "-ngl",
            "99",
            "--chunks",
            str(int(chunks)),
            "--kl-divergence",
            "--kl-divergence-base",
            str(logits_path),
        ],
        check=False,
        shell=False,
        capture_output=True,
    )
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    kld, _top1 = _parse_kl_output(output)
    if kld is None or result.returncode != 0:
        raise RuntimeError(
            f"EvoPress KL failed for {model_path.name}: {(output or 'no output')[:400]}"
        )
    return float(kld)


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except TypeError:
        if path.exists():
            path.unlink()


def _eval_config(
    block_types: Sequence[str],
    sources: Dict[str, Path],
    dataset_path: Path,
    logits_path: Path,
    chunks: int,
    cache: Dict[Tuple[Tuple[str, ...], int], float],
    stitch_cache: Dict[Tuple[str, ...], Path],
    tmp_dir: Path,
) -> float:
    cfg_key = tuple(block_types)
    key = (cfg_key, int(chunks))
    if key in cache:
        return cache[key]
    tmp_path = stitch_cache.get(cfg_key)
    if tmp_path is None or not tmp_path.is_file() or tmp_path.stat().st_size == 0:
        fd, tmp_name = tempfile.mkstemp(suffix=".gguf", prefix="evopress-", dir=str(tmp_dir))
        os.close(fd)
        tmp_path = Path(tmp_name)
        stitch_gguf(block_types, sources, tmp_path)
        stitch_cache[cfg_key] = tmp_path
    kld = _kl_vs_logits(tmp_path, dataset_path, logits_path, chunks)
    cache[key] = kld
    return kld


def evolutionary_search(
    sources: Dict[str, Path],
    fp16_path: Path,
    n_layers: int,
    *,
    target: float,
    tol: float,
    generations: int,
    offspring: int,
    survivors: Sequence[int],
    chunk_stages: Sequence[int],
    seed: int,
) -> Tuple[List[str], float]:
    from fiable.core.evaluate import _ensure_wikitext

    rng = random.Random(seed)
    dataset_path = _ensure_wikitext()
    tmp_dir = DATASETS_DIR / "evopress_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    logits_dir = DATASETS_DIR / "kl_logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    logits_path = logits_dir / f"evopress-{fp16_path.stem}.kld"
    max_chunks = max(int(c) for c in chunk_stages)
    console.print(f"[cyan]EvoPress: dumping FP16 logits ({max_chunks} chunks)...[/cyan]")
    _dump_logits(fp16_path, dataset_path, logits_path, max_chunks)

    parent = random_config(n_layers, rng, target, tol)
    cache: Dict[Tuple[Tuple[str, ...], int], float] = {}
    stitch_cache: Dict[Tuple[str, ...], Path] = {}
    try:
        parent_kl = _eval_config(
            parent,
            sources,
            dataset_path,
            logits_path,
            chunk_stages[0],
            cache,
            stitch_cache,
            tmp_dir,
        )
        console.print(
            f"[dim]EvoPress init mean bits {mean_bits(parent):.3f} KL {parent_kl:.6f}[/dim]"
        )

        stages = list(zip(survivors, chunk_stages))
        for gen in range(generations):
            pool = [list(parent)]
            for _ in range(offspring):
                child = mutate_config(parent, rng, target, tol)
                if child is not None:
                    pool.append(child)
            if len(pool) == 1:
                console.print("[yellow]EvoPress: no valid mutants this generation[/yellow]")
                continue
            ranked = pool
            last_kl = parent_kl
            for keep, chunks in stages:
                scored = []
                for i, cfg in enumerate(ranked, start=1):
                    console.print(
                        f"[dim]EvoPress gen {gen + 1}/{generations} "
                        f"candidate {i}/{len(ranked)} ({chunks} chunks)[/dim]"
                    )
                    kld = _eval_config(
                        cfg,
                        sources,
                        dataset_path,
                        logits_path,
                        chunks,
                        cache,
                        stitch_cache,
                        tmp_dir,
                    )
                    scored.append((kld, cfg))
                scored.sort(key=lambda item: item[0])
                ranked = [cfg for _kld, cfg in scored[: max(1, int(keep))]]
                last_kl = scored[0][0]
            parent = ranked[0]
            parent_kl = last_kl
            keep_parent = stitch_cache.get(tuple(parent))
            for cfg_key, path in list(stitch_cache.items()):
                if path != keep_parent:
                    _unlink(path)
                    stitch_cache.pop(cfg_key, None)
            console.print(
                f"[green]EvoPress gen {gen + 1}/{generations} "
                f"KL {parent_kl:.6f} bits {mean_bits(parent):.3f}[/green]"
            )
    finally:
        for path in stitch_cache.values():
            _unlink(path)
        _unlink(logits_path)
    return parent, parent_kl


def quantize_evopress(
    model: ModelConfig,
    quant_type: str,
    force: bool = False,
) -> "QuantizationResult":
    from fiable.core.quantize import QuantizationResult, convert_to_gguf, quantize_model
    from fiable.config.settings import ensure_llama_tools

    outfile = get_quantized_path(model, quant_type)
    meta_path = get_quant_meta_path(model, quant_type)
    if outfile.exists() and meta_path.exists() and not force:
        size = helpers.get_file_size_gb(outfile)
        console.print(
            f"[yellow]Skipping {quant_type} - already exists ({size:.2f} GB)[/yellow]"
        )
        return QuantizationResult(
            model_name=model.name,
            quantization_type=quant_type,
            success=True,
            output_path=outfile,
            file_size_gb=size,
            already_exists=True,
        )

    start = time.time()
    try:
        ensure_llama_tools(["llama-quantize", "llama-perplexity"])
        ok, fp16_path = convert_to_gguf(model, force=False)
        if not ok or fp16_path is None:
            raise RuntimeError("FP16 GGUF conversion failed; EvoPress needs it for KL")

        console.print("[cyan]EvoPress: ensuring uniform database GGUFs...[/cyan]")
        sources: Dict[str, Path] = {}
        for level in LEVELS:
            result = quantize_model(model, fp16_path, level, force=False)
            if not result.success or not result.output_path:
                raise RuntimeError(result.error or f"failed to build {level}")
            sources[level] = Path(result.output_path)

        gguf, _ = _import_gguf()
        template = gguf.GGUFReader(str(sources[NONBLOCK_SOURCE]))
        n_layers = n_blocks_from_tensors([t.name for t in template.tensors])
        del template

        generations = _env_int("FIABLE_EVOPRESS_GENERATIONS", 4)
        offspring = _env_int("FIABLE_EVOPRESS_OFFSPRING", 8)
        survivors = _env_int_list("FIABLE_EVOPRESS_SURVIVORS", (4, 1))
        chunk_stages = _env_int_list("FIABLE_EVOPRESS_CHUNKS", (1, 2))
        if len(chunk_stages) < len(survivors):
            chunk_stages = list(chunk_stages) + [chunk_stages[-1]] * (
                len(survivors) - len(chunk_stages)
            )
        chunk_stages = chunk_stages[: len(survivors)]
        target = _env_float("FIABLE_EVOPRESS_TARGET_BITS", 4.0)
        tol = _env_float("FIABLE_EVOPRESS_BITS_TOL", 0.05)
        seed = _env_int("FIABLE_EVOPRESS_SEED", 0)

        console.print(
            f"[cyan]EvoPress search: {n_layers} blocks, {generations}×{offspring} "
            f"target {target}±{tol} bits[/cyan]"
        )
        block_types, kl = evolutionary_search(
            sources,
            fp16_path,
            n_layers,
            target=target,
            tol=tol,
            generations=generations,
            offspring=offspring,
            survivors=survivors,
            chunk_stages=chunk_stages,
            seed=seed,
        )
        console.print(f"[cyan]Stitching {quant_type} GGUF...[/cyan]")
        stitch_gguf(block_types, sources, outfile)
        helpers.save_json(
            {
                "quantization_type": quant_type,
                "method": "evopress",
                "block_types": block_types,
                "target_bits": target,
                "bits_tol": tol,
                "mean_bits": round(mean_bits(block_types), 4),
                "generations": generations,
                "offspring": offspring,
                "survivors": list(survivors),
                "chunk_stages": list(chunk_stages),
                "kl": kl,
                "seed": seed,
                "gguf_path": str(outfile),
            },
            meta_path,
        )
        duration = time.time() - start
        size = helpers.get_file_size_gb(outfile)
        console.print(
            f"[green]✓ Created {quant_type} ({size:.2f} GB, mean bits "
            f"{mean_bits(block_types):.3f}, KL {kl:.6f}) "
            f"in {helpers.format_duration(duration)}[/green]"
        )
        return QuantizationResult(
            model_name=model.name,
            quantization_type=quant_type,
            success=True,
            output_path=outfile,
            file_size_gb=size,
            duration_seconds=duration,
        )
    except Exception as e:
        duration = time.time() - start
        console.print(f"[red]✗ EvoPress {quant_type} failed: {e}[/red]")
        return QuantizationResult(
            model_name=model.name,
            quantization_type=quant_type,
            success=False,
            error=str(e),
            duration_seconds=duration,
        )

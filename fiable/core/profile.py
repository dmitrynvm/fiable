"""Prefill vs decode profiling: wall-clock TTFT and inter-token latency."""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.table import Table

from fiable.config.settings import (
    DATASETS_DIR,
    LLAMA_BENCH,
    LLAMA_CLI,
    MODELS,
    REPORT_DIR,
    get_fp16_path,
    get_quantized_path,
)
from fiable.core.evaluate import _ensure_wikitext
from fiable.utils import helpers


console = Console()

PROMPT_TOKENS = 512
N_PREDICT = 128


@dataclass
class ProfileResult:
    model_name: str
    quantization_type: str
    model_path: str
    prompt_tokens: int = PROMPT_TOKENS
    n_predict: int = N_PREDICT
    ttft_ms: Optional[float] = None
    itl_ms: Optional[float] = None
    decode_tokens_per_sec: Optional[float] = None
    prefill_tokens_per_sec: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _prompt_file() -> Path:
    wiki = _ensure_wikitext()
    dest = DATASETS_DIR / "profile_prompt.txt"
    text = wiki.read_text(errors="ignore")
    # ~4 chars/token; overshoot so llama.cpp can use 512 prompt tokens
    dest.write_text(text[:2048])
    return dest


def _parse_perf(text: str) -> dict:
    """Parse llama-cli --perf lines and the Prompt/Generation summary footer."""
    out = {}
    prompt = re.search(
        r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens",
        text,
        re.IGNORECASE,
    )
    eval_ = re.search(
        r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)",
        text,
        re.IGNORECASE,
    )
    footer = re.search(
        r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s",
        text,
        re.IGNORECASE,
    )
    if prompt:
        ms, n = float(prompt.group(1)), int(prompt.group(2))
        out["prefill_ms"] = ms
        out["prefill_n"] = n
        if n and ms:
            out["prefill_tokens_per_sec"] = 1000.0 * n / ms
            out["ttft_ms"] = ms
    if eval_:
        ms, n = float(eval_.group(1)), int(eval_.group(2))
        out["decode_ms"] = ms
        out["decode_n"] = n
        if n and ms:
            out["itl_ms"] = ms / n
            out["decode_tokens_per_sec"] = 1000.0 * n / ms
    if footer:
        out.setdefault("prefill_tokens_per_sec", float(footer.group(1)))
        out.setdefault("decode_tokens_per_sec", float(footer.group(2)))
        dec = out.get("decode_tokens_per_sec")
        pref = out.get("prefill_tokens_per_sec")
        if dec and "itl_ms" not in out:
            out["itl_ms"] = 1000.0 / dec
        if pref and "ttft_ms" not in out:
            # prompt file is ~512 tokens by construction
            out["ttft_ms"] = 1000.0 * PROMPT_TOKENS / pref
    return out


def profile_generation(model_path: Path) -> ProfileResult:
    """Run llama-cli and measure wall-clock TTFT plus mean ITL from --perf."""
    stem = model_path.stem
    if stem.upper().endswith("-FP16") or stem.upper().endswith("_FP16"):
        quant = "FP16"
        name = stem[: -len("-fp16")]
    elif "-" in stem:
        quant = stem.split("-")[-1]
        # Q4_K_M is three dash-parts at the end
        for q in ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"):
            if stem.upper().endswith("-" + q):
                quant = q
                name = stem[: -(len(q) + 1)]
                break
        else:
            name = "-".join(stem.split("-")[:-1])
    else:
        quant = "unknown"
        name = stem

    result = ProfileResult(
        model_name=name,
        quantization_type=quant,
        model_path=str(model_path),
    )

    if not Path(LLAMA_CLI).exists():
        result.error = f"llama-cli not found: {LLAMA_CLI}"
        return result

    prompt_path = _prompt_file()
    cmd = [
        LLAMA_CLI,
        "-m",
        str(model_path),
        "-f",
        str(prompt_path),
        "-n",
        str(N_PREDICT),
        "-c",
        "2048",
        "-ngl",
        "99",
        "--no-display-prompt",
        "--simple-io",
        "--perf",
        "-no-cnv",
        "--single-turn",
    ]

    console.print(f"[cyan]Profiling {model_path.name} (TTFT + ITL)...[/cyan]")
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=600,
        )
        text = proc.stdout or ""
        perf = _parse_perf(text)
        result.ttft_ms = perf.get("ttft_ms")
        result.itl_ms = perf.get("itl_ms")
        result.decode_tokens_per_sec = perf.get("decode_tokens_per_sec")
        result.prefill_tokens_per_sec = perf.get("prefill_tokens_per_sec")
        if result.prefill_tokens_per_sec and result.ttft_ms is None:
            result.ttft_ms = 1000.0 * PROMPT_TOKENS / result.prefill_tokens_per_sec
        if result.decode_tokens_per_sec and result.itl_ms is None:
            result.itl_ms = 1000.0 / result.decode_tokens_per_sec
        if proc.returncode not in (0, None) and result.ttft_ms is None:
            result.error = (text or "llama-cli failed")[-500:]
        elif result.ttft_ms is not None:
            console.print(
                f"[green]✓ TTFT {result.ttft_ms:.1f} ms"
                + (f", ITL {result.itl_ms:.2f} ms/tok" if result.itl_ms else "")
                + "[/green]"
            )
        else:
            result.error = "No TTFT/ITL parsed from llama-cli output"
            console.print(f"[red]✗ {result.error}[/red]")
    except Exception as e:
        result.error = str(e)
        console.print(f"[red]✗ {e}[/red]")

    return result


def _default_profile_paths() -> List[Path]:
    paths = []
    for model in MODELS:
        fp16 = get_fp16_path(model)
        if fp16.exists():
            paths.append(fp16)
        q4 = get_quantized_path(model, "Q4_K_M")
        if q4.exists():
            paths.append(q4)
    return paths


def try_nsys(model_path: Path, tag: str) -> Optional[str]:
    """Best-effort Nsight wrap of a short llama-bench run. Returns path or error."""
    nsys = shutil.which("nsys")
    if not nsys:
        return None
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"nsys_{tag}"
    cmd = (
        f"{shlex.quote(nsys)} profile --trace=cuda --force-overwrite true "
        f"-o {shlex.quote(str(out))} "
        f"{shlex.quote(LLAMA_BENCH)} -m {shlex.quote(str(model_path))} "
        f"-p {PROMPT_TOKENS} -n {N_PREDICT} -ngl 999 -r 1 -o json"
    )
    console.print(f"[dim]nsys profile {model_path.name}...[/dim]")
    result = helpers.run_command(cmd, check=False)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()[:300]
        console.print(f"[yellow]nsys skipped: {err or 'non-zero exit'}[/yellow]")
        return f"nsys failed: {err}"
    console.print(f"[green]nsys report: {out}.nsys-rep[/green]")
    return str(out) + ".nsys-rep"


def profile_models(
    model_paths: Optional[List[Path]] = None,
    output_file: Optional[Path] = None,
    nsys: bool = True,
) -> List[ProfileResult]:
    """Profile FP16 vs Q4_K_M (or given paths) for TTFT and ITL."""
    if not model_paths:
        model_paths = _default_profile_paths()
    if not model_paths:
        console.print("[red]No FP16 or Q4_K_M models found in cache/[/red]")
        return []

    console.print(f"\n[bold]Profiling {len(model_paths)} model(s) (prefill vs decode)...[/bold]\n")
    results = []
    for path in model_paths:
        if not path.exists():
            console.print(f"[red]Not found: {path}[/red]")
            continue
        results.append(profile_generation(path))

    if nsys:
        fp16 = next((p for p in model_paths if p.exists() and "fp16" in p.stem.lower()), None)
        q4 = next((p for p in model_paths if p.exists() and "q4_k_m" in p.stem.lower()), None)
        if fp16:
            try_nsys(fp16, "fp16")
        if q4:
            try_nsys(q4, "q4")

    table = Table()
    table.add_column("Model", style="cyan")
    table.add_column("Quant", style="magenta")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("ITL (ms/tok)", justify="right")
    table.add_column("Prefill (tok/s)", justify="right")
    table.add_column("Decode (tok/s)", justify="right")
    for r in results:
        table.add_row(
            r.model_name,
            r.quantization_type,
            f"{r.ttft_ms:.1f}" if r.ttft_ms else "N/A",
            f"{r.itl_ms:.2f}" if r.itl_ms else "N/A",
            f"{r.prefill_tokens_per_sec:.1f}" if r.prefill_tokens_per_sec else "N/A",
            f"{r.decode_tokens_per_sec:.1f}" if r.decode_tokens_per_sec else "N/A",
        )
    console.print()
    console.print(table)

    if output_file is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = REPORT_DIR / f"profile_{stamp}.json"
    helpers.save_json([r.to_dict() for r in results], output_file)
    console.print(f"\n[green]Results saved to {output_file}[/green]")
    return results

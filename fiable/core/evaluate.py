"""Model evaluation: perplexity, benchmarks, throughput, and compression deltas."""

from __future__ import annotations

import fnmatch
import json
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from fiable.config.settings import (
    BENCHMARK_TASKS,
    DATASETS_DIR,
    REPORT_DIR,
    llama_binary,
    ensure_llama_tools,
    is_gptq_type,
    settings,
)
from fiable.core.metrics import (
    annotate_relative_metrics,
    default_eval_paths,
    parse_model_identity,
    peak_vram_sampler,
    pick_baseline,
    read_gguf_parameter_count,
)
from fiable.utils import helpers


console = Console()

WIKITEXT_ZIP_URL = (
    "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
)
WIKITEXT_MIN_BYTES = 100_000
GSM8K_TEST_SIZE = 1319
_PPL_CHUNK_RE = re.compile(rb"\[(\d+)\](\d+\.\d+),")
_PPL_TOTAL_RE = re.compile(
    rb"(?:calculating perplexity|computing) over (\d+) chunks", re.IGNORECASE
)
_TQDM_RE = re.compile(rb"(\d+)\s*/\s*(\d+)")
_KEEP_PPL_LOG = re.compile(
    r"seconds per pass|Final estimate|tokeniz|failed|error|ETA ", re.I
)


def _eval_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("[green]{task.fields[detail]}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        refresh_per_second=4,
        expand=True,
    )


def _split_incomplete_ppl(buf: bytes) -> tuple[bytes, bytes]:
    for i in range(1, min(24, len(buf)) + 1):
        tail = buf[-i:]
        if tail.startswith(b"[") and re.fullmatch(rb"\[\d*\]?\d*\.?\d*", tail):
            return buf[:-i], tail
    return buf, b""


class _PerplexityEchoFilter:
    """Drive a Rich bar from llama-perplexity chunk dumps; hide `[n]ppl,` spam."""

    def __init__(self, progress: Progress, task_id: int, total: Optional[int] = None) -> None:
        self.progress = progress
        self.task_id = task_id
        self._carry = b""
        self._log_carry = ""
        if total:
            self.progress.update(self.task_id, total=total)

    def _emit_logs(self, text: str) -> None:
        self._log_carry += text
        while "\n" in self._log_carry:
            line, self._log_carry = self._log_carry.split("\n", 1)
            stripped = line.strip()
            if stripped and _KEEP_PPL_LOG.search(stripped):
                self.progress.console.print(f"[dim]  {stripped}[/dim]")

    def __call__(self, data: bytes) -> bytes:
        complete, self._carry = _split_incomplete_ppl(self._carry + data)
        for match in _PPL_TOTAL_RE.finditer(complete):
            self.progress.update(self.task_id, total=int(match.group(1)))
        last_end = 0
        last_n = None
        last_ppl = None
        logs = bytearray()
        for match in _PPL_CHUNK_RE.finditer(complete):
            logs.extend(complete[last_end : match.start()])
            last_n = int(match.group(1))
            last_ppl = float(match.group(2))
            last_end = match.end()
        logs.extend(complete[last_end:])
        if last_n is not None:
            self.progress.update(
                self.task_id,
                completed=last_n,
                description="perplexity",
                detail=f"PPL {last_ppl:.2f}",
            )
        self._emit_logs(logs.decode("utf-8", errors="replace"))
        return b""

    def flush(self) -> bytes:
        leftover = self._carry
        self._carry = b""
        if leftover:
            self._emit_logs(leftover.decode("utf-8", errors="replace"))
        if self._log_carry.strip() and _KEEP_PPL_LOG.search(self._log_carry):
            self.progress.console.print(f"[dim]  {self._log_carry.strip()}[/dim]")
        self._log_carry = ""
        return b""


class _LmEvalEchoFilter:
    """Parse lm-eval / tqdm `n/total` into a Rich bar; keep a few INFO lines."""

    def __init__(self, progress: Progress, task_id: int, total: Optional[int] = None) -> None:
        self.progress = progress
        self.task_id = task_id
        self._buf = b""
        if total:
            self.progress.update(self.task_id, total=total)

    def _handle_piece(self, piece: bytes) -> None:
        text = piece.decode("utf-8", errors="replace").strip()
        if not text:
            return
        match = _TQDM_RE.search(piece)
        if match:
            done, total = int(match.group(1)), int(match.group(2))
            if total > 0:
                self.progress.update(self.task_id, total=total, completed=done, detail="")
            return
        lowered = text.lower()
        if "running generate_until" in lowered:
            self.progress.update(self.task_id, description="gsm8k generate", detail="generating")
            self.progress.console.print(f"[dim]  {text}[/dim]")
            return
        if "building contexts" in lowered:
            self.progress.update(self.task_id, description="gsm8k setup", detail="building contexts")
            self.progress.console.print(f"[dim]  {text}[/dim]")
            return
        if "error" in lowered or "failed" in lowered or "traceback" in lowered:
            self.progress.console.print(f"[red]  {text}[/red]")

    def __call__(self, data: bytes) -> bytes:
        self._buf += data.replace(b"\r", b"\n")
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            self._handle_piece(line)
        return b""

    def flush(self) -> bytes:
        if self._buf:
            self._handle_piece(self._buf)
            self._buf = b""
        return b""


@dataclass
class EvaluationResult:
    """Complete evaluation result for a model."""
    model_path: str
    model_name: str
    quantization_type: str
    file_size_gb: float
    n_params: Optional[int] = None
    bits_per_weight: Optional[float] = None
    compression_ratio: Optional[float] = None
    baseline_quant: Optional[str] = None
    perplexity: Optional[float] = None
    perplexity_error: Optional[str] = None
    perplexity_long: Optional[float] = None
    perplexity_long_error: Optional[str] = None
    delta_ppl: Optional[float] = None
    benchmarks: Dict[str, float] = field(default_factory=dict)
    benchmark_errors: Dict[str, str] = field(default_factory=dict)
    delta_acc: Dict[str, float] = field(default_factory=dict)
    acc_retention: Dict[str, float] = field(default_factory=dict)
    size_reduction: Optional[float] = None
    efficiency_score: Dict[str, float] = field(default_factory=dict)
    throughput_tokens_per_sec: Optional[float] = None
    throughput_latency_ms: Optional[float] = None
    throughput_stddev: Optional[float] = None
    throughput_memory_gb: Optional[float] = None
    peak_vram_gb: Optional[float] = None
    throughput_error: Optional[str] = None
    prefill_tokens_per_sec: Optional[float] = None
    ttft_ms: Optional[float] = None
    speedup: Optional[float] = None
    prefill_speedup: Optional[float] = None
    kl_divergence: Optional[float] = None
    top1_match: Optional[float] = None
    kl_divergence_error: Optional[str] = None
    timestamp: str = field(default_factory=helpers.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _ensure_wikitext(split: str = "test") -> Path:
    """Download WikiText-2 raw split if missing. Raises on failure."""
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"Unknown WikiText split: {split}")
    filename = f"wiki.{split}.raw"
    text_path = DATASETS_DIR / filename
    if text_path.exists() and text_path.stat().st_size >= WIKITEXT_MIN_BYTES:
        return text_path

    text_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = text_path.parent / "wikitext-2-raw-v1.zip"
    if not zip_path.exists():
        console.print(f"[cyan]Downloading WikiText-2 ({filename})...[/cyan]")
        urllib.request.urlretrieve(WIKITEXT_ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        member = next(
            (n for n in zf.namelist() if n.endswith(filename)),
            None,
        )
        if member is None:
            raise FileNotFoundError(f"{filename} not found in WikiText zip")
        with zf.open(member) as src, open(text_path, "wb") as dst:
            dst.write(src.read())

    if text_path.stat().st_size < WIKITEXT_MIN_BYTES:
        raise ValueError(f"WikiText file looks too small: {text_path}")
    return text_path


def _parse_perplexity(stdout: str) -> Optional[float]:
    finals = re.findall(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)", stdout, re.IGNORECASE)
    if finals:
        return float(finals[-1])
    for pattern in (
        r"perplexity:\s*([0-9.]+)",
        r"\bPPL\s*=\s*([0-9.]+)",
    ):
        matches = re.findall(pattern, stdout, re.IGNORECASE)
        if matches:
            return float(matches[-1])
    return None


def _parse_bench_rows(stdout: str) -> tuple[Optional[dict], Optional[dict]]:
    """Return (prefill/pp row, decode/tg row) from llama-bench JSON stdout."""
    start = stdout.find("[")
    end = stdout.rfind("]")
    if start < 0 or end <= start:
        return None, None
    rows = json.loads(stdout[start : end + 1])
    if not rows:
        return None, None
    prefill = next(
        (r for r in rows if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0),
        None,
    )
    decode = next((r for r in reversed(rows) if r.get("n_gen", 0) > 0), None)
    return prefill, decode


def _parse_bench_json(stdout: str) -> Optional[dict]:
    """Pick the token-generation row from llama-bench JSON stdout."""
    _prefill, decode = _parse_bench_rows(stdout)
    return decode


def _run_stream_with_ppl_bar(
    cmd: str,
    description: str,
    total: Optional[int] = None,
) -> subprocess.CompletedProcess:
    with _eval_progress() as progress:
        task_id = progress.add_task(description, total=total, detail="")
        return helpers.run_command(
            cmd,
            check=False,
            stream=True,
            echo_filter=_PerplexityEchoFilter(progress, task_id, total=total),
        )


def evaluate_perplexity(
    model_path: Path,
    dataset: str = "wikitext",
    context_size: int = 512,
    chunks: Optional[int] = None,
) -> tuple[Optional[float], Optional[str]]:
    """Evaluate model perplexity on a dataset with llama-perplexity."""
    label = f"{dataset} ctx={context_size}"
    console.print(f"[cyan]Evaluating perplexity ({label})...[/cyan]")

    ppl_bin = llama_binary("llama-perplexity")
    if not ppl_bin.is_file():
        return None, f"llama-perplexity binary not found: {ppl_bin}"

    try:
        if dataset != "wikitext":
            return None, f"Unsupported dataset '{dataset}' (only wikitext is implemented)"
        dataset_path = _ensure_wikitext()

        cmd = (
            f"{shlex.quote(str(ppl_bin))} "
            f"-m {shlex.quote(str(model_path))} "
            f"-f {shlex.quote(str(dataset_path))} "
            f"-c {context_size} -ngl 99"
        )
        if chunks is not None:
            cmd += f" --chunks {int(chunks)}"
        started = time.time()
        result = _run_stream_with_ppl_bar(cmd, f"perplexity ctx={context_size}", total=chunks)
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        console.print(f"[dim]  elapsed {helpers.format_duration(time.time() - started)}[/dim]")

        if result.returncode != 0:
            err = (result.stderr or result.stdout or "unknown error").strip()
            return None, f"Perplexity command failed: {err[:500]}"

        ppl = _parse_perplexity(output)
        if ppl is None:
            return None, "Failed to parse perplexity from llama-perplexity output"

        console.print(f"[green]✓ Perplexity ({label}): {ppl:.2f}[/green]")
        return ppl, None

    except Exception as e:
        return None, str(e)


def _extract_lm_eval_score(metrics: dict) -> Optional[float]:
    prefer = (
        "acc,none",
        "acc_norm,none",
        "exact_match,strict-match",
        "exact_match,none",
        "pass@1,none",
        "acc",
        "acc_norm",
        "exact_match",
        "pass@1",
    )
    for key in prefer:
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    for key, val in metrics.items():
        if not isinstance(val, (int, float)):
            continue
        lowered = key.lower()
        if "stderr" in lowered or "alias" in lowered:
            continue
        return float(val)
    return None


def _score_for_logical_task(results_map: dict, logical: str, patterns: List[str]) -> Optional[float]:
    if logical in results_map:
        score = _extract_lm_eval_score(results_map[logical])
        if score is not None:
            return score
    matched: List[float] = []
    for key, metrics in results_map.items():
        if not isinstance(metrics, dict):
            continue
        for pat in patterns:
            if fnmatch.fnmatch(key, pat) or key == logical or key.startswith(logical + "_"):
                score = _extract_lm_eval_score(metrics)
                if score is not None:
                    matched.append(score)
                break
    if matched:
        return sum(matched) / len(matched)
    return None


def _find_lm_eval_results(out_dir: Path) -> Optional[dict]:
    direct = out_dir / "results.json"
    candidates = []
    if direct.exists():
        candidates.append(direct)
    candidates.extend(
        sorted(out_dir.rglob("results*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    )
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, dict) and "results" in data:
            return data
    return None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_http(url: str, timeout: float = 180.0, label: str = "llama-server") -> bool:
    deadline = time.time() + timeout
    started = time.time()
    last_print = 0.0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    console.print(
                        f"[dim]  {label} ready in {helpers.format_duration(time.time() - started)}[/dim]"
                    )
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            now = time.time()
            if now - last_print >= 5:
                console.print(
                    f"[dim]  Waiting for {label} ({helpers.format_duration(now - started)})...[/dim]"
                )
                last_print = now
            time.sleep(1)
    return False


def _start_llama_server(model_path: Path) -> tuple[subprocess.Popen, int]:
    server_bin = llama_binary("llama-server")
    if not server_bin.is_file():
        raise FileNotFoundError(f"llama-server binary not found: {server_bin}")
    port = _free_port()
    cmd = [
        str(server_bin),
        "-m",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-ngl",
        "99",
        "-c",
        str(settings.LM_EVAL_MAX_LENGTH),
        "--alias",
        "fiable-eval",
    ]
    console.print(
        f"[dim]  Starting llama-server {model_path.name} on 127.0.0.1:{port} "
        f"(ctx={settings.LM_EVAL_MAX_LENGTH}, ngl=99)...[/dim]"
    )
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )
    ready = _wait_http(f"http://127.0.0.1:{port}/health") or _wait_http(
        f"http://127.0.0.1:{port}/v1/models"
    )
    if not ready:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise TimeoutError("llama-server did not become ready")
    return proc, port


def evaluate_benchmarks(
    model_path: Path,
    tasks: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> tuple[Dict[str, float], Dict[str, str]]:
    """Evaluate GGUF models with lm-eval against a local llama-server."""
    if tasks is None:
        tasks = ["gsm8k"]

    limit_note = f"limit={limit} examples/task" if limit else "full dataset"
    console.print(
        f"[cyan]Running benchmarks: {', '.join(tasks)} ({limit_note}, batch_size=1)...[/cyan]"
    )
    if not limit:
        if "gsm8k" in tasks:
            console.print(
                f"[dim]  GSM8K test set is {GSM8K_TEST_SIZE} problems; "
                "each generates a solution.[/dim]"
            )
        else:
            console.print("[dim]  Full task set.[/dim]")

    scores: Dict[str, float] = {}
    errors: Dict[str, str] = {}

    lm_eval_bin = shutil.which("lm_eval")
    if lm_eval_bin is None:
        msg = "lm-eval not installed; skipped (no placeholder scores). Install: pip install lm-eval"
        console.print(f"[yellow]{msg}[/yellow]")
        for task in tasks:
            errors[task] = "lm-eval not installed"
        return scores, errors

    proc = None
    try:
        proc, port = _start_llama_server(model_path)
        with tempfile.TemporaryDirectory(prefix="fiable-lm-eval-") as tmp:
            out_dir = Path(tmp)
            lm_tasks: List[str] = []
            for task in tasks:
                lm_tasks.extend(BENCHMARK_TASKS.get(task, [task]))
            cmd = (
                f"{shlex.quote(lm_eval_bin)} --model gguf "
                f"--model_args base_url=http://127.0.0.1:{port},"
                f"max_length={settings.LM_EVAL_MAX_LENGTH} "
                f"--tasks {shlex.quote(','.join(lm_tasks))} "
                f"--output_path {shlex.quote(str(out_dir))} "
                f"--batch_size 1"
            )
            if limit:
                cmd += f" --limit {int(limit)}"
            if limit:
                bar_total = int(limit) * max(len(lm_tasks), 1)
            elif tasks == ["gsm8k"] or lm_tasks == ["gsm8k"]:
                bar_total = GSM8K_TEST_SIZE
            else:
                bar_total = None
            started = time.time()
            with _eval_progress() as progress:
                task_id = progress.add_task(
                    "lm-eval", total=bar_total, detail="starting"
                )
                result = helpers.run_command(
                    cmd,
                    check=False,
                    stream=True,
                    echo_filter=_LmEvalEchoFilter(progress, task_id, total=bar_total),
                )
            console.print(
                f"[dim]  lm-eval finished in {helpers.format_duration(time.time() - started)}[/dim]"
            )
            if result.returncode != 0:
                err = (result.stderr or result.stdout or "unknown error").strip()
                for task in tasks:
                    errors[task] = f"Benchmark failed: {err[:500]}"
                    console.print(f"[red]  {task}: failed[/red]")
                return scores, errors

            payload = _find_lm_eval_results(out_dir)
            if not payload:
                for task in tasks:
                    errors[task] = "lm-eval produced no results.json"
                    console.print(f"[red]  {task}: no results.json[/red]")
                return scores, errors

            results_map = payload.get("results", {})
            for task in tasks:
                patterns = BENCHMARK_TASKS.get(task, [task])
                score = _score_for_logical_task(results_map, task, patterns)
                if score is None:
                    errors[task] = f"No accuracy field in lm-eval results for {task}"
                    console.print(f"[red]  {task}: no score field[/red]")
                    continue
                scores[task] = score
                console.print(f"[green]  {task}: {score:.3f}[/green]")
            return scores, errors
    except Exception as e:
        for task in tasks:
            errors.setdefault(task, str(e))
            console.print(f"[red]  {task}: error - {e}[/red]")
        return scores, errors
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()


def evaluate_throughput(
    model_path: Path,
    prompt_length: int = 512,
    n_gen: int = 128,
) -> dict:
    """Evaluate prefill (pp) and decode (tg) with llama-bench; sample peak VRAM via NVML."""
    console.print("[cyan]Benchmarking prefill (pp) and decode (tg)...[/cyan]")
    empty = {
        "prefill_tokens_per_sec": None,
        "ttft_ms": None,
        "decode_tokens_per_sec": None,
        "decode_latency_ms": None,
        "decode_stddev": None,
        "n_params": None,
        "memory_gb": None,
        "error": None,
    }

    bench_bin = llama_binary("llama-bench")
    if not bench_bin.is_file():
        empty["error"] = f"llama-bench binary not found: {bench_bin}"
        return empty

    try:
        cmd = (
            f"{shlex.quote(str(bench_bin))} "
            f"-m {shlex.quote(str(model_path))} "
            f"-p {prompt_length} -n {n_gen} "
            f"-ngl 999 -o json -r 3"
        )
        console.print("[dim]  llama-bench 3 reps (pp + tg), live output below[/dim]")
        started = time.time()
        with peak_vram_sampler() as vram:
            result = helpers.run_command(cmd, check=False, stream=True)
        console.print(f"[dim]  elapsed {helpers.format_duration(time.time() - started)}[/dim]")
        if vram.get("peak_vram_gb") is not None:
            empty["memory_gb"] = vram["peak_vram_gb"]

        if result.returncode != 0:
            err = (result.stderr or result.stdout or "unknown error").strip()
            empty["error"] = f"llama-bench failed: {err[:500]}"
            return empty

        prefill, decode = _parse_bench_rows(result.stdout or "")
        row = decode or prefill or {}
        if row.get("model_n_params"):
            empty["n_params"] = int(row["model_n_params"])
        if prefill and prefill.get("avg_ts"):
            empty["prefill_tokens_per_sec"] = float(prefill["avg_ts"])
            if empty["prefill_tokens_per_sec"]:
                n_prompt = int(prefill.get("n_prompt") or prompt_length)
                empty["ttft_ms"] = 1000.0 * n_prompt / empty["prefill_tokens_per_sec"]
        if decode and decode.get("avg_ts"):
            empty["decode_tokens_per_sec"] = float(decode["avg_ts"])
            if empty["decode_tokens_per_sec"]:
                empty["decode_latency_ms"] = 1000.0 / empty["decode_tokens_per_sec"]
            if decode.get("stddev_ts") is not None:
                empty["decode_stddev"] = float(decode["stddev_ts"])
            if empty["memory_gb"] is None and decode.get("vram_used_mib") is not None:
                empty["memory_gb"] = float(decode["vram_used_mib"]) / 1024.0

        if empty["prefill_tokens_per_sec"] is None and empty["decode_tokens_per_sec"] is None:
            empty["error"] = "Failed to parse llama-bench JSON (no pp/tg avg_ts)"
            return empty

        parts = []
        if empty["prefill_tokens_per_sec"] is not None:
            parts.append(f"Prefill {empty['prefill_tokens_per_sec']:.1f} tok/s")
        if empty["ttft_ms"] is not None:
            parts.append(f"TTFT {empty['ttft_ms']:.1f} ms")
        if empty["decode_tokens_per_sec"] is not None:
            parts.append(f"Decode {empty['decode_tokens_per_sec']:.1f} tok/s")
        if empty["decode_latency_ms"] is not None:
            parts.append(f"Latency {empty['decode_latency_ms']:.1f} ms/tok")
        if empty["memory_gb"] is not None:
            parts.append(f"VRAM {empty['memory_gb']:.2f} GB")
        console.print(f"[green]✓ {', '.join(parts)}[/green]")
        return empty

    except Exception as e:
        empty["error"] = str(e)
        return empty


def _parse_kl_output(text: str) -> tuple[Optional[float], Optional[float]]:
    kld = None
    top1 = None
    match = re.search(r"Mean\s+KLD:\s*([0-9.]+)", text)
    if match:
        kld = float(match.group(1))
    match = re.search(r"Same top p:\s*([0-9.]+)", text)
    if match:
        top1 = float(match.group(1)) / 100.0
    return kld, top1


def evaluate_kl_vs_baseline(results: List[EvaluationResult], chunks: Optional[int] = None) -> None:
    """Dump FP16 logits once per family, then KL-diverge each quant against them."""
    if chunks is None:
        chunks = settings.KL_CHUNKS
    ppl_bin = llama_binary("llama-perplexity")
    if not ppl_bin.is_file():
        for result in results:
            result.kl_divergence_error = f"llama-perplexity binary not found: {ppl_bin}"
        return

    try:
        dataset_path = _ensure_wikitext()
    except Exception as e:
        for result in results:
            result.kl_divergence_error = str(e)
        return

    by_model: Dict[str, List[EvaluationResult]] = {}
    for result in results:
        by_model.setdefault(result.model_name, []).append(result)

    logits_dir = DATASETS_DIR / "kl_logits"
    logits_dir.mkdir(parents=True, exist_ok=True)

    for model_name, group in by_model.items():
        baseline = pick_baseline(group)
        if baseline is None:
            continue
        logits_path = logits_dir / f"{model_name}.kld"
        console.print(
            f"[cyan]Dumping baseline logits for KL ({model_name} {baseline.quantization_type})...[/cyan]"
        )
        dump = _run_stream_with_ppl_bar(
            f"{shlex.quote(str(ppl_bin))} "
            f"-m {shlex.quote(baseline.model_path)} "
            f"-f {shlex.quote(str(dataset_path))} "
            f"-c 512 -ngl 99 --chunks {int(chunks)} "
            f"--save-all-logits {shlex.quote(str(logits_path))}",
            "KL baseline logits",
            total=int(chunks),
        )
        if dump.returncode != 0 or not logits_path.exists():
            err = (dump.stderr or dump.stdout or "failed to dump logits").strip()[:400]
            for result in group:
                result.kl_divergence_error = f"KL baseline dump failed: {err}"
            continue

        for result in group:
            if result.model_path == baseline.model_path:
                result.kl_divergence = 0.0
                result.top1_match = 1.0
                console.print(f"[green]✓ KL {result.quantization_type}: 0.0000 (baseline)[/green]")
                continue
            console.print(f"[cyan]KL divergence {result.quantization_type} vs {baseline.quantization_type}...[/cyan]")
            cmp_ = _run_stream_with_ppl_bar(
                f"{shlex.quote(str(ppl_bin))} "
                f"-m {shlex.quote(result.model_path)} "
                f"-f {shlex.quote(str(dataset_path))} "
                f"-c 512 -ngl 99 --chunks {int(chunks)} "
                f"--kl-divergence --kl-divergence-base {shlex.quote(str(logits_path))}",
                f"KL {result.quantization_type}",
                total=int(chunks),
            )
            output = (cmp_.stdout or "") + "\n" + (cmp_.stderr or "")
            kld, top1 = _parse_kl_output(output)
            if kld is None:
                result.kl_divergence_error = "Failed to parse Mean KLD from llama-perplexity"
                console.print(f"[red]✗ KL {result.quantization_type}: parse failed[/red]")
                continue
            result.kl_divergence = kld
            result.top1_match = top1
            extra = f", top-1 {top1:.1%}" if top1 is not None else ""
            console.print(f"[green]✓ KL {result.quantization_type}: {kld:.6f}{extra}[/green]")

        try:
            logits_path.unlink(missing_ok=True)
        except TypeError:
            if logits_path.exists():
                logits_path.unlink()


def evaluate_model(
    model_path: Path,
    run_perplexity: bool = True,
    run_benchmarks: bool = True,
    run_throughput: bool = True,
    run_long_context: bool = True,
    dataset: str = "wikitext",
    benchmark_tasks: Optional[List[str]] = None,
    limit: Optional[int] = None,
    index: Optional[int] = None,
    total: Optional[int] = None,
) -> EvaluationResult:
    """Run evaluation on a single model (relative metrics filled later)."""
    prefix = f"[{index}/{total}] " if index is not None and total is not None else ""
    console.print(f"\n[bold cyan]{prefix}Evaluating {model_path.name}...[/bold cyan]")

    model_name, quant_type = parse_model_identity(model_path)
    from fiable.core.gptq import effective_size_gb

    file_size = effective_size_gb(model_path)
    n_params = read_gguf_parameter_count(model_path)

    result = EvaluationResult(
        model_path=str(model_path),
        model_name=model_name,
        quantization_type=quant_type,
        file_size_gb=file_size,
        n_params=n_params,
    )

    if run_perplexity:
        ppl, ppl_error = evaluate_perplexity(model_path, dataset, chunks=limit)
        result.perplexity = ppl
        result.perplexity_error = ppl_error

    if run_long_context:
        long_ppl, long_err = evaluate_perplexity(
            model_path,
            dataset,
            context_size=settings.LONG_CONTEXT_SIZE,
            chunks=limit,
        )
        result.perplexity_long = long_ppl
        result.perplexity_long_error = long_err

    if run_benchmarks:
        scores, errors = evaluate_benchmarks(model_path, benchmark_tasks, limit=limit)
        result.benchmarks = scores
        result.benchmark_errors = errors

    if run_throughput:
        if is_gptq_type(quant_type):
            console.print(
                "[yellow]Skipping llama-bench for GPTQ "
                "(GGUF is dequant F16; kernels would not be GPTQ)[/yellow]"
            )
            result.throughput_error = "skipped: GPTQ GGUF is reconstructed F16, not GPTQ kernels"
        else:
            bench = evaluate_throughput(model_path)
            result.prefill_tokens_per_sec = bench["prefill_tokens_per_sec"]
            result.ttft_ms = bench["ttft_ms"]
            result.throughput_tokens_per_sec = bench["decode_tokens_per_sec"]
            result.throughput_latency_ms = bench["decode_latency_ms"]
            result.throughput_stddev = bench["decode_stddev"]
            result.throughput_memory_gb = bench["memory_gb"]
            result.peak_vram_gb = bench["memory_gb"]
            result.throughput_error = bench["error"]
            if bench.get("n_params") and not result.n_params:
                result.n_params = bench["n_params"]

    if index is not None and total is not None:
        console.print(f"[dim]{prefix}{model_path.name} complete ({total - index} remaining)[/dim]")

    return result


def evaluate_models(
    model_paths: List[Path],
    run_perplexity: bool = True,
    run_benchmarks: bool = True,
    run_throughput: bool = True,
    run_long_context: bool = True,
    run_kl: bool = True,
    dataset: str = "wikitext",
    benchmark_tasks: Optional[List[str]] = None,
    limit: Optional[int] = None,
    benchmark_limit: Optional[int] = None,
    output_file: Optional[Path] = None,
) -> List[EvaluationResult]:
    """Evaluate multiple models and attach FP16-relative compression metrics."""
    if limit is None:
        limit = benchmark_limit
    needed = []
    if run_perplexity or run_long_context or run_kl:
        needed.append("llama-perplexity")
    if run_throughput:
        needed.append("llama-bench")
    if run_benchmarks:
        needed.append("llama-server")
    if needed:
        console.print("[dim]Ensuring llama.cpp eval tools are available...[/dim]")
        try:
            tools = ensure_llama_tools(needed)
            for name, path in tools.items():
                console.print(f"[dim]  {name}: {path}[/dim]")
        except Exception as e:
            console.print(f"[red]Failed to build llama.cpp tools: {e}[/red]")
            return []

    console.print(f"\n[bold]Evaluating {len(model_paths)} model(s)...[/bold]")
    sample_note = f"{limit} samples" if limit else "full datasets"
    console.print(f"[dim]Sample cap: {sample_note} (PPL chunks + accuracy examples)[/dim]")
    per_model = []
    if run_perplexity:
        per_model.append("WikiText PPL ctx=512")
    if run_long_context:
        per_model.append(f"WikiText PPL ctx={settings.LONG_CONTEXT_SIZE}")
    if run_benchmarks:
        names = ",".join(benchmark_tasks or ["gsm8k"])
        per_model.append(names)
    if run_throughput:
        per_model.append("llama-bench pp/tg")
    if per_model:
        console.print(f"[dim]Per model: {' → '.join(per_model)}[/dim]")
    if run_kl:
        console.print("[dim]After all models: KL vs FP16 baseline[/dim]")
    console.print()

    results = []
    n_models = len(model_paths)
    for i, model_path in enumerate(model_paths, 1):
        if not model_path.exists():
            console.print(f"[red][{i}/{n_models}] Model not found: {model_path}[/red]")
            continue
        started = time.time()
        results.append(
            evaluate_model(
                model_path,
                run_perplexity=run_perplexity,
                run_benchmarks=run_benchmarks,
                run_throughput=run_throughput,
                run_long_context=run_long_context,
                dataset=dataset,
                benchmark_tasks=benchmark_tasks,
                limit=limit,
                index=i,
                total=n_models,
            )
        )
        console.print(
            f"[dim][{i}/{n_models}] wall time {helpers.format_duration(time.time() - started)}[/dim]"
        )

    annotate_relative_metrics(results)
    if run_kl and results:
        evaluate_kl_vs_baseline(results, chunks=limit)

    print_evaluation_summary(results)

    if output_file is None:
        output_file = REPORT_DIR / "evaluation.json"

    helpers.save_json(build_evaluation_report(results), output_file)
    console.print(f"\n[green]Results saved to {output_file}[/green]")
    return results


def _round(value: Optional[float], digits: int) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def summary_row(result: EvaluationResult) -> Dict[str, Any]:
    """Compact comparison row for the evaluation report summary."""
    row: Dict[str, Any] = {
        "model": result.model_name,
        "quant": result.quantization_type,
        "baseline": result.baseline_quant,
        "size_gb": _round(result.file_size_gb, 3),
        "compression_ratio": _round(result.compression_ratio, 3),
        "size_reduction": _round(result.size_reduction, 4),
        "bits_per_weight": _round(result.bits_per_weight, 3),
        "perplexity": _round(result.perplexity, 4),
        "delta_ppl": _round(result.delta_ppl, 4),
        "perplexity_long": _round(result.perplexity_long, 4),
        "kl_divergence": _round(result.kl_divergence, 4),
        "top1_match": _round(result.top1_match, 4),
        "prefill_tok_s": _round(result.prefill_tokens_per_sec, 2),
        "decode_tok_s": _round(result.throughput_tokens_per_sec, 2),
        "latency_ms": _round(result.throughput_latency_ms, 2),
        "speedup": _round(result.speedup, 3),
        "prefill_speedup": _round(result.prefill_speedup, 3),
        "vram_gb": _round(result.peak_vram_gb or result.throughput_memory_gb, 2),
    }
    for task, score in (result.benchmarks or {}).items():
        row[f"acc_{task}"] = _round(score, 4)
    for task, delta in (result.delta_acc or {}).items():
        row[f"delta_acc_{task}"] = _round(delta, 4)
    for task, ret in (result.acc_retention or {}).items():
        row[f"acc_retention_{task}"] = _round(ret, 4)
    for task, eff in (result.efficiency_score or {}).items():
        row[f"efficiency_{task}"] = _round(eff, 4)
    return {k: v for k, v in row.items() if v is not None}


def build_evaluation_report(results: List[EvaluationResult]) -> Dict[str, Any]:
    """Single report document: summary table plus full per-model results."""
    return {
        "generated_at": helpers.timestamp(),
        "n_models": len(results),
        "summary": [summary_row(r) for r in results],
        "results": [r.to_dict() for r in results],
    }


def print_evaluation_summary(results: List[EvaluationResult]) -> None:
    """Print a summary table of evaluation results."""
    if not results:
        return

    acc_tasks: List[str] = []
    for result in results:
        for task in list(result.benchmarks) + list(result.benchmark_errors):
            if task not in acc_tasks:
                acc_tasks.append(task)

    table = Table()
    table.add_column("Model", style="cyan")
    table.add_column("Quant", style="magenta")
    table.add_column("Size", justify="right")
    table.add_column("Ratio", justify="right")
    table.add_column("bpw", justify="right")
    table.add_column("PPL", justify="right")
    table.add_column("ΔPPL", justify="right")
    table.add_column("PPL-long", justify="right")
    table.add_column("KL", justify="right")
    for task in acc_tasks:
        table.add_column(task.upper(), justify="right")
        table.add_column(f"Δ{task.upper()}", justify="right")
        table.add_column(f"Ret {task.upper()}", justify="right")
        table.add_column(f"Eff {task.upper()}", justify="right")
    table.add_column("Decode", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("VRAM", justify="right")

    def _fmt(val: Optional[float], spec: str) -> str:
        return format(val, spec) if val is not None else "N/A"

    for result in results:
        cells = [
            result.model_name,
            result.quantization_type,
            _fmt(result.file_size_gb, ".2f"),
            _fmt(result.compression_ratio, ".2f") + ("x" if result.compression_ratio else ""),
            _fmt(result.bits_per_weight, ".2f"),
            _fmt(result.perplexity, ".2f"),
            _fmt(result.delta_ppl, "+.1%") if result.delta_ppl is not None else "N/A",
            _fmt(result.perplexity_long, ".2f"),
            _fmt(result.kl_divergence, ".4f"),
        ]
        for task in acc_tasks:
            acc = (result.benchmarks or {}).get(task)
            cells.append(_fmt(acc, ".1%") if acc is not None else "N/A")
            delta = (result.delta_acc or {}).get(task)
            cells.append(_fmt(delta, "+.1%") if delta is not None else "N/A")
            retention = (result.acc_retention or {}).get(task)
            cells.append(_fmt(retention, ".1%") if retention is not None else "N/A")
            efficiency = (result.efficiency_score or {}).get(task)
            cells.append(_fmt(efficiency, ".1%") if efficiency is not None else "N/A")
        cells.append(_fmt(result.throughput_tokens_per_sec, ".1f"))
        cells.append(
            _fmt(result.speedup, ".2f") + ("x" if result.speedup is not None else "")
        )
        cells.append(_fmt(result.peak_vram_gb or result.throughput_memory_gb, ".2f"))
        table.add_row(*cells)

    console.print()
    console.print(table)

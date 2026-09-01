"""Configuration and constants for the compression pipeline."""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


# Artifacts default to ./store and ./report in the process cwd.
# Override with FIABLE_HOME, FIABLE_STORE_DIR, or FIABLE_REPORT_DIR.
_ROOT = _env_path("FIABLE_HOME", Path.cwd())


def _resolve_store_dir() -> Path:
    """./store, migrating a legacy ./cache directory when present."""
    if os.environ.get("FIABLE_STORE_DIR"):
        return _env_path("FIABLE_STORE_DIR", _ROOT / "store")
    store = (_ROOT / "store").resolve()
    legacy = (_ROOT / "cache").resolve()
    if not store.exists() and legacy.exists() and legacy.is_dir():
        try:
            legacy.rename(store)
        except OSError:
            return legacy
    return store


class Settings:
    """Global settings for Fiable."""
    
    # Directory paths
    STORE_DIR = _resolve_store_dir()
    OUTPUT_DIR = STORE_DIR  # quantized GGUFs live next to FP16 in store/
    REPORT_DIR = _env_path("FIABLE_REPORT_DIR", _ROOT / "report")
    CHARTS_DIR = REPORT_DIR  # PNGs live next to evaluation.json
    DATASETS_DIR = STORE_DIR / "datasets"

    LONG_CONTEXT_SIZE = 2048
    KL_CHUNKS = 1
    LM_EVAL_MAX_LENGTH = 2048

    # Evaluation settings
    EVAL_DATASETS = {
        "wikitext": "wikitext-2-raw-v1",
        "ptb": "ptb_text_only",
    }

    BENCHMARK_TASKS = {
        "mmlu": ["mmlu"],
        "gsm8k": ["gsm8k"],
        "humaneval": ["humaneval"],
    }

    # Visualization: matplotlib tab10 (blue dashed / red x / green square)
    CHART_DPI = 300
    CHART_STYLE = "whitegrid"
    CHART_PALETTE = "tab10"
    CHART_COLORS = [
        "#1f77b4",  # tab:blue
        "#d62728",  # tab:red
        "#2ca02c",  # tab:green
        "#ff7f0e",  # tab:orange
        "#9467bd",  # tab:purple
        "#8c564b",  # tab:brown
    ]

    # Public llama.cpp type. EvoPress still builds Q2_K…Q6_K internally as a search DB.
    QUANT_TYPES: List[str] = ["Q4_K_M"]
    GPTQ_TYPES: List[str] = ["GPTQ_4"]
    EVOPRESS_TYPES: List[str] = ["EVOPRESS_4"]
    DEFAULT_QUANT_TYPES: List[str] = ["Q4_K_M", "GPTQ_4", "EVOPRESS_4"]

    # Available models
    MODELS: List["ModelConfig"] = []


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    repo: str
    local_dir: str
    fp16_filename: str
    quant_prefix: str
    
    def __post_init__(self):
        """Ensure paths are Path objects."""
        if isinstance(self.local_dir, str):
            self.local_dir = Path(self.local_dir)
        if isinstance(self.fp16_filename, str):
            self.fp16_filename = Path(self.fp16_filename)


# Create global settings instance
settings = Settings()

# Configure available models
settings.MODELS = [
    ModelConfig(
        name="Llama 3.1 8B",
        repo="meta-llama/Meta-Llama-3.1-8B-Instruct",
        local_dir="Meta-Llama-3.1-8B-Instruct",
        fp16_filename="llama-3.1-8b-instruct-fp16.gguf",
        quant_prefix="llama-3.1-8b-instruct",
    ),
    ModelConfig(
        name="Qwen 2.5 7B",
        repo="Qwen/Qwen2.5-7B-Instruct",
        local_dir="Qwen2.5-7B-Instruct",
        fp16_filename="qwen2.5-7b-instruct-fp16.gguf",
        quant_prefix="qwen2.5-7b-instruct",
    ),
]

# Export constants for backward compatibility
STORE_DIR = settings.STORE_DIR
DATASETS_DIR = settings.DATASETS_DIR
OUTPUT_DIR = settings.OUTPUT_DIR
REPORT_DIR = settings.REPORT_DIR
CHARTS_DIR = settings.CHARTS_DIR
# Legacy names (deprecated)
BASE_DIR = settings.STORE_DIR
COMPRESSED_DIR = settings.OUTPUT_DIR
RESULTS_DIR = settings.REPORT_DIR
EVAL_DATASETS = settings.EVAL_DATASETS
BENCHMARK_TASKS = settings.BENCHMARK_TASKS
CHART_DPI = settings.CHART_DPI
CHART_STYLE = settings.CHART_STYLE
CHART_PALETTE = settings.CHART_PALETTE
CHART_COLORS = settings.CHART_COLORS
QUANT_TYPES = settings.QUANT_TYPES
GPTQ_TYPES = settings.GPTQ_TYPES
EVOPRESS_TYPES = settings.EVOPRESS_TYPES
DEFAULT_QUANT_TYPES = settings.DEFAULT_QUANT_TYPES
MODELS = settings.MODELS

_GPTQ_ALIASES = {
    "GPTQ": "GPTQ_4",
    "GPTQ4": "GPTQ_4",
    "GPTQ_4": "GPTQ_4",
}
_GPTQ_TO_WA = {
    "GPTQ_4": "W4A16 g128",
}
_EVOPRESS_ALIASES = {
    "EVOPRESS": "EVOPRESS_4",
    "EVOPRESS4": "EVOPRESS_4",
    "EVOPRESS_4": "EVOPRESS_4",
}
_EVOPRESS_TO_WA = {
    "EVOPRESS_4": "W4A16 evo",
}

# GGUF is weight-only; activations stay F16 at runtime (W*A16).
_GGUF_TO_WA = {
    "FP16": "W16A16",
    "F16": "W16A16",
    "BF16": "W16A16",
    "FP32": "W32A32",
    "F32": "W32A32",
    "Q4_K_M": "W4A16",
}
_WA_TO_GGUF = {
    "W16A16": "FP16",
    "W32A32": "FP32",
    "W4A16": "Q4_K_M",
}
_PUBLIC_GGUF = {"Q4_K_M", "FP16", "F16", "BF16", "FP32", "F32"}
_WA_RE = re.compile(r"^W(\d+)A(\d+)(?:G(\d+))?$", re.IGNORECASE)


def is_gptq_type(quant: str) -> bool:
    """True for GPTQ_* methods (not llama-quantize GGUF types)."""
    key = (quant or "").strip().upper().replace("-", "_").replace(" ", "")
    return key in _GPTQ_ALIASES or key.startswith("GPTQ")


def is_evopress_type(quant: str) -> bool:
    """True for EVOPRESS_* mixed-GGUF search methods."""
    key = (quant or "").strip().upper().replace("-", "_").replace(" ", "")
    return key in _EVOPRESS_ALIASES or key.startswith("EVOPRESS")


def format_precision(quant: str) -> str:
    """Q4_K_M → W4A16. Unknown names are returned unchanged."""
    key = (quant or "").strip().upper()
    compact = key.replace("-", "_").replace(" ", "")
    if key in _GPTQ_TO_WA:
        return _GPTQ_TO_WA[key]
    alias = _GPTQ_ALIASES.get(compact)
    if alias and alias in _GPTQ_TO_WA:
        return _GPTQ_TO_WA[alias]
    if key in _EVOPRESS_TO_WA:
        return _EVOPRESS_TO_WA[key]
    evo = _EVOPRESS_ALIASES.get(compact)
    if evo and evo in _EVOPRESS_TO_WA:
        return _EVOPRESS_TO_WA[evo]
    if key in _GGUF_TO_WA:
        return _GGUF_TO_WA[key]
    compact = key.replace(" ", "")
    match = _WA_RE.match(compact)
    if match:
        label = f"W{int(match.group(1))}A{int(match.group(2))}"
        if match.group(3):
            label += f" g{int(match.group(3))}"
        return label
    return quant


def parse_quant_spec(spec: str) -> str:
    """W4A16 or Q4_K_M → llama.cpp GGUF type. GPTQ_4 / EVOPRESS_4 aliases."""
    raw = (spec or "").strip()
    if not raw:
        raise ValueError("Empty precision")
    key = raw.upper().replace("-", "_").replace(" ", "")
    if key in _GPTQ_ALIASES:
        return _GPTQ_ALIASES[key]
    if key in _EVOPRESS_ALIASES:
        return _EVOPRESS_ALIASES[key]
    match = _WA_RE.match(raw.upper().replace(" ", ""))
    if match:
        wa = f"W{int(match.group(1))}A{int(match.group(2))}"
        gguf = _WA_TO_GGUF.get(wa)
        if gguf is None:
            known = ", ".join(_WA_TO_GGUF)
            raise ValueError(
                f"Unknown precision '{raw}'. Use Q4_K_M, GPTQ_4, EVOPRESS_4, or: {known}"
            )
        act = int(match.group(2))
        if act not in (16, 32):
            extra = " Try W4A16 (Q4_K_M)."
            raise ValueError(
                f"'{raw}' needs {act}-bit activations; GGUF quants are weight-only (A16).{extra}"
            )
        return gguf
    if key in _PUBLIC_GGUF:
        return "FP16" if key in {"F16"} else key
    raise ValueError(
        f"Unknown method '{raw}'. Supported: Q4_K_M, GPTQ_4, EVOPRESS_4 "
        f"(aliases: W4A16, GPTQ, EVOPRESS)."
    )


# Keep Hugging Face datasets under store/
os.environ.setdefault("HF_DATASETS_CACHE", str(settings.DATASETS_DIR))

LLAMA_TOOLS = (
    "llama-quantize",
    "llama-perplexity",
    "llama-bench",
    "llama-cli",
    "llama-server",
)
_OPT_LLAMA_ROOT = Path("/opt/llama.cpp")


def llama_src_dir() -> Path:
    return settings.STORE_DIR / "llama.cpp"


def llama_bin_dir() -> Path:
    return llama_src_dir() / "build" / "bin"


def _opt_llama_dir() -> Optional[Path]:
    """Prebuilt CUDA llama.cpp (Vast image: /opt/llama.cpp/cuda-<ver>)."""
    if not _OPT_LLAMA_ROOT.is_dir():
        return None
    pinned = os.environ.get("FIABLE_LLAMA_DIR")
    if pinned:
        path = Path(pinned).expanduser()
        return path if path.is_dir() else None
    cuda_dirs = sorted(
        (p for p in _OPT_LLAMA_ROOT.glob("cuda-*") if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    return cuda_dirs[0] if cuda_dirs else None


def _has_cuda_backend(bin_dir: Path) -> bool:
    return bin_dir.is_dir() and any(bin_dir.glob("libggml-cuda*"))


def llama_binary(name: str) -> Path:
    """Resolve a llama.cpp tool: env, PATH, store build, then /opt."""
    env_key = "FIABLE_" + name.upper().replace("-", "_")
    override = os.environ.get(env_key)
    if override:
        return Path(override).expanduser().resolve()
    found = shutil.which(name)
    if found:
        return Path(found).resolve()
    built = llama_bin_dir() / name
    if built.is_file():
        return built
    opt_dir = _opt_llama_dir()
    if opt_dir is not None:
        opt = opt_dir / name
        if opt.is_file():
            return opt
    return built


def ensure_llama_src() -> Path:
    """Clone llama.cpp into store/ if needed. Returns the source directory."""
    src = llama_src_dir()
    if not src.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/ggerganov/llama.cpp.git",
                str(src),
            ],
            check=True,
        )
    return src


def _nvcc_path() -> Optional[Path]:
    found = shutil.which("nvcc")
    if found:
        return Path(found)
    for candidate in (
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        "/usr/local/cuda",
        "/usr/local/cuda-12.8",
        "/usr/local/cuda-12.4",
        "/usr/local/cuda-12",
        "/opt/cuda",
    ):
        if not candidate:
            continue
        nvcc = Path(candidate) / "bin" / "nvcc"
        if nvcc.is_file():
            return nvcc
    return None


def _want_cuda() -> bool:
    if os.environ.get("FIABLE_NO_CUDA", "").lower() in {"1", "true", "yes"}:
        return False
    return _nvcc_path() is not None


def _cuda_apt_release() -> Optional[str]:
    nvcc = _nvcc_path()
    if nvcc is None:
        return None
    try:
        out = subprocess.check_output([str(nvcc), "--version"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"release (\d+)\.(\d+)", out)
    if not match:
        return None
    return f"{match.group(1)}-{match.group(2)}"


def ensure_cublas_dev() -> None:
    """Install libcublas-dev so CMake's CUDA::cublas target exists."""
    cuda_home = Path(os.environ.get("CUDA_HOME") or "/usr/local/cuda")
    if (cuda_home / "include" / "cublas.h").is_file() and (
        cuda_home / "lib64" / "libcublas.so"
    ).exists():
        return
    script = Path(__file__).resolve().parents[2] / "scripts" / "install-cuda-dev.sh"
    if script.is_file():
        subprocess.run(["bash", str(script)], check=True)
        return
    release = _cuda_apt_release()
    if not release:
        return
    pkg = f"libcublas-dev-{release}"
    apt = ["apt-get", "install", "-y", pkg]
    if os.geteuid() != 0:
        apt = ["sudo", *apt]
    subprocess.run(["apt-get", "update", "-qq"], check=False)
    subprocess.run(apt, check=True)


def ensure_llama_tools(names: Optional[List[str]] = None) -> Dict[str, Path]:
    """Build missing llama.cpp tools into store/llama.cpp/build/bin."""
    names = list(names or LLAMA_TOOLS)
    force_build = os.environ.get("FIABLE_BUILD_LLAMA", "").lower() in {"1", "true", "yes"}
    resolved = {name: llama_binary(name) for name in names}
    have_all = all(path.is_file() for path in resolved.values())
    want_cuda = _want_cuda()
    backend_dirs = {path.parent for path in resolved.values() if path.is_file()}
    backend_dirs.add(llama_bin_dir())
    opt_dir = _opt_llama_dir()
    if opt_dir is not None:
        backend_dirs.add(opt_dir)
    cuda_backend = any(_has_cuda_backend(d) for d in backend_dirs)
    if have_all and not force_build and (not want_cuda or cuda_backend):
        return resolved

    src = ensure_llama_src()
    if want_cuda:
        ensure_cublas_dev()

    build_dir = src / "build"
    cmake_args = [
        "cmake",
        "-S",
        str(src),
        "-B",
        str(build_dir),
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DGGML_CCACHE=OFF",
    ]
    nvcc = _nvcc_path()
    if nvcc is not None:
        cmake_args.append("-DGGML_CUDA=ON")
        cmake_args.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")
        cuda_bin = str(nvcc.parent)
        env = os.environ.copy()
        env["PATH"] = cuda_bin + os.pathsep + env.get("PATH", "")
    else:
        env = None
    subprocess.run(cmake_args, check=True, env=env)

    jobs = str(os.cpu_count() or 4)
    subprocess.run(
        ["cmake", "--build", str(build_dir), "-j", jobs, "--target", *names],
        check=True,
        env=env,
    )

    resolved = {name: llama_binary(name) for name in names}
    missing = [name for name, path in resolved.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "llama.cpp tools missing after build: "
            + ", ".join(f"{n} ({resolved[n]})" for n in missing)
        )
    return resolved


def get_model_by_name(name: str) -> ModelConfig:
    """Get model configuration by name."""
    for model in settings.MODELS:
        if model.name == name:
            return model
    raise ValueError(f"Model '{name}' not found. Available: {[m.name for m in settings.MODELS]}")


def get_fp16_path(model: ModelConfig) -> Path:
    """Get path to FP16 GGUF file for a model."""
    return settings.STORE_DIR / model.fp16_filename


def get_quantized_path(model: ModelConfig, quant_type: str) -> Path:
    """Get path to quantized model file."""
    return settings.OUTPUT_DIR / f"{model.quant_prefix}-{quant_type}.gguf"


def get_quant_meta_path(model: ModelConfig, quant_type: str) -> Path:
    """Sidecar JSON next to a GGUF (packed GPTQ size, bits, group size)."""
    return settings.OUTPUT_DIR / f"{model.quant_prefix}-{quant_type}.meta.json"


def get_gptq_packed_dir(model: ModelConfig, quant_type: str) -> Path:
    """Hugging Face GPTQ checkpoint directory."""
    return settings.OUTPUT_DIR / f"{model.quant_prefix}-{quant_type}"

"""Configuration and constants for the compression pipeline."""

import os
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

    # Quantization types
    QUANT_TYPES: List[str] = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]

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
MODELS = settings.MODELS

# Keep Hugging Face datasets under store/
os.environ.setdefault("HF_DATASETS_CACHE", str(settings.DATASETS_DIR))

LLAMA_TOOLS = (
    "llama-quantize",
    "llama-perplexity",
    "llama-bench",
    "llama-cli",
    "llama-server",
)
_OPT_LLAMA_DIR = Path("/opt/llama.cpp/cuda-12.8")


def llama_src_dir() -> Path:
    return settings.STORE_DIR / "llama.cpp"


def llama_bin_dir() -> Path:
    return llama_src_dir() / "build" / "bin"


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
    opt = _OPT_LLAMA_DIR / name
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


def _want_cuda() -> bool:
    if os.environ.get("FIABLE_NO_CUDA", "").lower() in {"1", "true", "yes"}:
        return False
    if shutil.which("nvcc"):
        return True
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home and (Path(cuda_home) / "bin" / "nvcc").is_file():
        return True
    return False


def ensure_llama_tools(names: Optional[List[str]] = None) -> Dict[str, Path]:
    """Build missing llama.cpp tools into store/llama.cpp/build/bin."""
    names = list(names or LLAMA_TOOLS)
    src = ensure_llama_src()
    resolved = {name: llama_binary(name) for name in names}
    if all(path.is_file() for path in resolved.values()):
        return resolved

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
        "-DGGML_CCACHE=OFF",
    ]
    if _want_cuda():
        cmake_args.append("-DGGML_CUDA=ON")
    subprocess.run(cmake_args, check=True)

    jobs = str(os.cpu_count() or 4)
    build = subprocess.run(
        ["cmake", "--build", str(build_dir), "-j", jobs, "--target", *names],
    )
    if build.returncode != 0:
        subprocess.run(
            ["cmake", "--build", str(build_dir), "-j", jobs, "--target", "llama-quantize"],
            check=True,
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

"""Configuration and constants for the compression pipeline."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any


class Settings:
    """Global settings for Fiable."""
    
    # Directory paths
    CACHE_DIR = Path("/workspace/cache")
    OUTPUT_DIR = CACHE_DIR  # quantized GGUFs live next to FP16 in cache/
    REPORT_DIR = Path("/workspace/report")
    CHARTS_DIR = REPORT_DIR  # PNGs live next to evaluation.json
    DATASETS_DIR = CACHE_DIR / "datasets"

    # llama.cpp binaries
    LLAMA_QUANTIZE = "/opt/llama.cpp/cuda-12.8/llama-quantize"
    LLAMA_PERPLEXITY = "/opt/llama.cpp/cuda-12.8/llama-perplexity"
    LLAMA_BENCH = "/opt/llama.cpp/cuda-12.8/llama-bench"
    LLAMA_CLI = "/opt/llama.cpp/cuda-12.8/llama-cli"
    LLAMA_SERVER = "/opt/llama.cpp/cuda-12.8/llama-server"

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
CACHE_DIR = settings.CACHE_DIR
DATASETS_DIR = settings.DATASETS_DIR
OUTPUT_DIR = settings.OUTPUT_DIR
REPORT_DIR = settings.REPORT_DIR
CHARTS_DIR = settings.CHARTS_DIR
# Legacy names (deprecated)
BASE_DIR = settings.CACHE_DIR
COMPRESSED_DIR = settings.OUTPUT_DIR
RESULTS_DIR = settings.REPORT_DIR
LLAMA_QUANTIZE = settings.LLAMA_QUANTIZE
LLAMA_PERPLEXITY = settings.LLAMA_PERPLEXITY
LLAMA_BENCH = settings.LLAMA_BENCH
LLAMA_CLI = settings.LLAMA_CLI
LLAMA_SERVER = settings.LLAMA_SERVER
EVAL_DATASETS = settings.EVAL_DATASETS
BENCHMARK_TASKS = settings.BENCHMARK_TASKS
CHART_DPI = settings.CHART_DPI
CHART_STYLE = settings.CHART_STYLE
CHART_PALETTE = settings.CHART_PALETTE
CHART_COLORS = settings.CHART_COLORS
QUANT_TYPES = settings.QUANT_TYPES
MODELS = settings.MODELS

# Keep Hugging Face datasets under cache/
os.environ.setdefault("HF_DATASETS_CACHE", str(settings.DATASETS_DIR))


def get_model_by_name(name: str) -> ModelConfig:
    """Get model configuration by name."""
    for model in settings.MODELS:
        if model.name == name:
            return model
    raise ValueError(f"Model '{name}' not found. Available: {[m.name for m in settings.MODELS]}")


def get_fp16_path(model: ModelConfig) -> Path:
    """Get path to FP16 GGUF file for a model."""
    return settings.CACHE_DIR / model.fp16_filename


def get_quantized_path(model: ModelConfig, quant_type: str) -> Path:
    """Get path to quantized model file."""
    return settings.OUTPUT_DIR / f"{model.quant_prefix}-{quant_type}.gguf"

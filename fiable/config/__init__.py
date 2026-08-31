"""Configuration module for Fiable."""

from fiable.config.settings import (
    ModelConfig,
    MODELS,
    QUANT_TYPES,
    STORE_DIR,
    DATASETS_DIR,
    OUTPUT_DIR,
    REPORT_DIR,
    CHARTS_DIR,
    # Legacy names (deprecated)
    BASE_DIR,
    COMPRESSED_DIR,
    RESULTS_DIR,
    EVAL_DATASETS,
    BENCHMARK_TASKS,
    CHART_DPI,
    CHART_STYLE,
    CHART_PALETTE,
    CHART_COLORS,
    get_model_by_name,
    get_fp16_path,
    get_quantized_path,
    llama_binary,
    ensure_llama_src,
    ensure_llama_tools,
    settings,
)

__all__ = [
    "ModelConfig",
    "MODELS",
    "QUANT_TYPES",
    "STORE_DIR",
    "DATASETS_DIR",
    "OUTPUT_DIR",
    "REPORT_DIR",
    "CHARTS_DIR",
    "BASE_DIR",  # deprecated
    "COMPRESSED_DIR",  # deprecated
    "RESULTS_DIR",  # deprecated
    "EVAL_DATASETS",
    "BENCHMARK_TASKS",
    "CHART_DPI",
    "CHART_STYLE",
    "CHART_PALETTE",
    "CHART_COLORS",
    "get_model_by_name",
    "get_fp16_path",
    "get_quantized_path",
    "llama_binary",
    "ensure_llama_src",
    "ensure_llama_tools",
    "settings",
]

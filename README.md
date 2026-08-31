# Fiable - Model Compression Tool

A comprehensive CLI tool for downloading, quantizing, evaluating, and visualizing large language model compression using llama.cpp.

## Quick Start

```bash
# Navigate to project
cd fiable

# Install
pip install -e .

# Use
fiable
fiable cache
fiable download
```

---

## 📦 Project Structure

This follows the standard Python project layout:

```
/workspace/fiable/              ← Project directory (work here!)
├── fiable/                     ← Package code
│   ├── cli/                    ← Command-line interface
│   │   └── commands.py         ← All CLI commands
│   ├── config/                 ← Configuration
│   │   └── settings.py         ← Settings & ModelConfig
│   ├── core/                   ← Core business logic
│   │   ├── download.py         ← HuggingFace downloads
│   │   ├── quantize.py         ← GGUF quantization
│   │   ├── evaluate.py         ← Model evaluation
│   │   ├── metrics.py          ← Compression deltas, NVML, Pareto
│   │   └── profile.py          ← Prefill vs decode profiling
│   ├── utils/                  ← Utilities
│   │   └── helpers.py          ← Helper functions
│   └── visual/                 ← Charts & plots
│       └── plots.py            ← Chart generation
├── setup.py                    ← Installation script
├── pyproject.toml              ← Modern packaging config
├── requirements.txt            ← Dependencies
└── README.md                   ← This file
```

## 📖 Usage

### Typical Workflow

```bash
# 1. List cache (downloads, quants, datasets)
fiable cache

# 2. Download models
fiable download

# 3. Quantize models
fiable quantize --types "Q4_K_M,Q5_K_M"

# 4. Evaluate models (defaults to all)
fiable evaluate

# 5. Prefill vs decode (FP16 vs Q4_K_M)
fiable profile --no-nsys -o report/exp2_profile.json

# 6. View results in report/
```

### Download Models

```bash
# Download all configured models
fiable download

# Download specific models
fiable download "Llama 3.1 8B" "Qwen 2.5 7B"

# Force re-download
fiable download --force
```

### Quantize Models

```bash
# Quantize all downloaded models
fiable quantize

# Quantize with specific types
fiable quantize "Llama 3.1 8B" --types "Q4_K_M,Q5_K_M"

# Force re-quantization
fiable quantize --force
```

### Evaluate Models

```bash
# Evaluate all models in cache/ → report/evaluation.json
fiable evaluate

# Evaluate specific models
fiable evaluate cache/*Q4_K_M.gguf
```

# Evaluate with custom options
fiable evaluate cache/llama-3.1-8b-instruct-Q4_K_M.gguf --no-benchmarks --dataset wikitext
```

### Generate Visualizations

```bash
# Generate charts from the evaluation report (writes PNGs to report/)
fiable plot
```

### List Information

```bash
fiable cache  # Each artifact: type, size, file
```

---

## 🐍 Python API

```python
# Import modules
from fiable.core import download, quantize, evaluate
from fiable.config import settings
from fiable.utils import helpers
from fiable.visual import plots

# Download models
results = download.download_models()

# Quantize models
quant_results = quantize.quantize_models()

# Evaluate models
eval_results = evaluate.evaluate_models(model_paths)

# Generate charts
plots.generate_all_charts(results_file)
```

---

## ⚙️ Configuration

Edit `fiable/config/settings.py` to customize:

### Add Models

```python
from fiable.config.settings import ModelConfig, settings

settings.MODELS.append(ModelConfig(
    name="Your Model",
    repo="org/model-name",
    local_dir="local-directory",
    fp16_filename="model-fp16.gguf",
    quant_prefix="model-prefix",
))
```

### Modify Quantization Types

```python
settings.QUANT_TYPES = ["Q4_K_M", "Q5_K_M", "Q8_0"]
```

### Configure Paths

```python
settings.CACHE_DIR = Path("/custom/cache")
settings.REPORT_DIR = Path("/custom/report")
```

---

## 📊 Models & Quantization

### Configured Models

1. **Llama 3.1 8B Instruct** - Excellent for general chat and English tasks
2. **Qwen 2.5 7B Instruct** - Superior for code generation and math reasoning

---

## 🎨 CLI Commands Reference

### Download Command
```bash
fiable download [MODEL_NAMES...] [OPTIONS]
```

**Options:**
- `--force, -f` - Force re-download even if exists

### Quantize Command
```bash
fiable quantize [MODEL_NAMES...] [OPTIONS]
```

**Options:**
- `--types, -t TEXT` - Comma-separated quantization types
- `--force, -f` - Force re-quantization even if exists

### Evaluate Command
```bash
fiable evaluate [MODEL_PATHS...] [OPTIONS]
```

By default, evaluates **all GGUF files in `cache/`** (FP16 baselines and quants).

**Options:**
- `MODEL_PATHS` - Optional: specific GGUF paths (default: cache/*.gguf)
- `--perplexity/--no-perplexity` - WikiText-2 perplexity at ctx=512 (default: true)
- `--long-context/--no-long-context` - Perplexity at ctx=2048 (default: true)
- `--benchmarks/--no-benchmarks` - lm-eval via llama-server (default: true)
- `--throughput/--no-throughput` - llama-bench + NVML peak VRAM (default: true)
- `--kl/--no-kl` - KL divergence vs FP16/Q8 logits (default: true)
- `--dataset, -d TEXT` - Dataset for perplexity
- `--tasks, -t TEXT` - Comma-separated benchmark tasks (default: mmlu,gsm8k)
- `--limit INTEGER` - lm-eval example cap per task (omit for full run)
- `--output, -o PATH` - Output JSON (default: `report/evaluation.json`)

### Plot Command
```bash
fiable plot [RESULTS_FILE] [OPTIONS]
```

Defaults to `report/evaluation.json`.

**Options:**
- `--output-dir, -o PATH` - Output directory for charts (default: `report/`)

### Profile Command (prefill vs decode)
```bash
fiable profile [MODEL_PATHS...] [OPTIONS]
```

Defaults to `cache/*fp16.gguf` and `cache/*Q4_K_M.gguf`. Measures time-to-first-token (prefill) and inter-token latency (decode).

```bash
fiable profile
fiable profile --no-nsys -o report/exp2_profile.json
fiable evaluate cache/*fp16.gguf cache/*Q4_K_M.gguf --no-benchmarks -o report/exp2_bench.json
```

**Prefill vs decode:** prefill is usually compute-bound (Q4 can be slower than FP16 because of dequant). Decode is usually memory-bound (Q4 is often faster because 4-bit weights move less data from HBM).

**Options:**
- `--nsys/--no-nsys` - Try Nsight Systems on a short llama-bench if `nsys` is installed
- `--output, -o PATH` - Output JSON

---

## 📁 Output Files

### Directory Structure
```
/workspace/
├── cache/                      # FP16, quantized GGUFs, datasets, downloads
│   └── datasets/               # WikiText and Hugging Face datasets
├── report/                     # evaluation.json, chart PNGs + CSVs
│   └── evaluation.json
```

### Evaluation report (`report/evaluation.json`)
One file: a compact `summary` table plus full `results`.

```json
{
  "generated_at": "2026-08-31T08:42:00",
  "n_models": 2,
  "summary": [
    {
      "model": "llama-3.1-8b-instruct",
      "quant": "Q4_K_M",
      "baseline": "FP16",
      "size_gb": 4.58,
      "compression_ratio": 3.26,
      "bits_per_weight": 4.92,
      "perplexity": 7.53,
      "delta_ppl": 0.028,
      "kl_divergence": 0.031,
      "decode_tok_s": 262.4,
      "vram_gb": 5.1
    }
  ],
  "results": [{ "...": "full per-model metrics" }]
}
```

Relative fields (`compression_ratio`, `delta_ppl`, `delta_acc`, `kl_divergence`) are vs the family's FP16 baseline, or Q8_0 if FP16 is missing.

### Generated Charts
Each chart in `report/` has a PNG and a CSV of the plotted points:
- `perplexity_vs_size.png` / `.csv` — Perplexity vs Size
- `throughput.png` / `.csv` — Throughput
- `latency.png` / `.csv` — Latency (ms/tok)
- `compression.png` / `.csv` — Compression
- `perplexity_vs_throughput.png` / `.csv` — Perplexity vs Throughput
- `accuracy.png` / `.csv` — Accuracy vs Size

---

## 🔧 Development

### Running from Source
```bash
# Without installation
cd /workspace/fiable
python -m fiable --help

# Check imports
python -c "import fiable; print(fiable.__version__)"
```

### Adding Custom Evaluation Datasets
```python
# In fiable/config/settings.py
settings.EVAL_DATASETS = {
    "wikitext": "wikitext-2-raw-v1",
    "custom": "path/to/dataset",
}
```

### Module Structure

- **`cli/`** - User interface and command definitions
- **`config/`** - Configuration management and settings
- **`core/`** - Business logic (download, quantize, evaluate)
- **`utils/`** - Shared utility functions
- **`visual/`** - Chart generation and plotting

---

## 🐛 Troubleshooting

### Authentication Issues
```bash
# Set HuggingFace token
export HF_TOKEN=your_token_here

# Or login via CLI
huggingface-cli login
```

### Missing Dependencies
```bash
# Reinstall
cd /workspace/fiable
pip install -e .
```

### Disk Space Issues
- Process one model at a time
- Delete original models after conversion
- Only keep needed quantization types

```bash
fiable download "Llama 3.1 8B"
fiable quantize "Llama 3.1 8B" --types "Q4_K_M"
```

### Import Errors
```bash
# Verify installation
cd /workspace/fiable
pip install -e .
python -c "import fiable; print('OK')"
```

---

## 📝 Examples

### Complete Workflow

```bash
# Start from scratch
cd /workspace/fiable

# 1. Check configured models
fiable cache

# 2. Download a specific model
fiable download "Llama 3.1 8B"

# 3. Quantize to recommended format
fiable quantize "Llama 3.1 8B" --types "Q4_K_M"

# 4. Evaluate all quantized models
fiable evaluate

# 5. Generate visualization
fiable plot
```

### Batch Evaluation
```bash
# Evaluate all models (writes report/evaluation.json)
fiable evaluate

# Evaluate all Q4_K_M models into the same report
fiable evaluate cache/*Q4_K_M.gguf

# Evaluate a model family
fiable evaluate cache/llama*.gguf
```

### Generate Charts from the Report
```bash
fiable plot
```

---

## 🏗️ Architecture Benefits

### Modularity
- Clear separation between CLI, config, core logic, utils, and visualization
- Each module has a single, focused responsibility

### Scalability
- Easy to add new modules or features
- Can split large files into sub-modules
- Clear growth path for the codebase

### Maintainability
- Intuitive file organization
- Clear where to add new features
- Logical import paths

### Testability
- Easy to test individual modules
- Clear dependency boundaries
- Simple mocking for unit tests

### Professional
- Follows Python packaging best practices
- Standard project structure (like requests/, flask/, django/)
- Modern packaging with pyproject.toml

---

## 📦 Package Info

- **Name:** fiable
- **Version:** 0.1.0
- **Python:** 3.8+
- **License:** MIT
- **Entry Point:** `fiable` (command-line)

### Dependencies
- typer>=0.12.0 - CLI framework
- rich>=13.7.0 - Terminal formatting
- huggingface_hub>=0.23.0 - Model downloads
- matplotlib>=3.8.0 - Chart generation
- seaborn>=0.13.0 - Statistical visualization
- pandas>=2.0.0 - Data manipulation
- lm-eval>=0.4.0 - MMLU / GSM8K / HumanEval harness
- datasets>=2.14.0 - Hugging Face datasets (WikiText, benchmarks)
- evaluate>=0.4.0 - Hugging Face evaluate metrics
- transformers>=4.40.0 - Model/tokenizer support for lm-eval
- accelerate>=0.26.0 - Hugging Face accelerate
- llama-cpp-python>=0.2.0 - Optional GGUF Python bindings
- nvidia-ml-py>=12.0.0 - NVML peak VRAM during llama-bench

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 🙏 Credits

Built with:
- [Typer](https://typer.tiangolo.com/) - CLI framework
- [Rich](https://rich.readthedocs.io/) - Terminal formatting
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Model quantization
- [Hugging Face](https://huggingface.co/) - Model hub
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - Benchmark evaluation

---

## 🚀 Getting Started Checklist

- [ ] Navigate to project: `cd /workspace/fiable`
- [ ] Install package: `pip install -e .`
- [ ] Test CLI: `fiable --help`
- [ ] List cache: `fiable cache`
- [ ] Configure models in `fiable/config/settings.py`
- [ ] Download models: `fiable download`
- [ ] View results in `report/`

---

**Ready to compress some models?**

```bash
cd /workspace/fiable
fiable cache    # See cached artifacts
fiable download     # Start downloading
```

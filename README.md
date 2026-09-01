# Fiable - Model Compression Tool

A comprehensive CLI tool for downloading, quantizing, evaluating, and visualizing large language model compression using llama.cpp.

## Quick Start

```bash
git clone git@github.com:dmitrynvm/fiable.git
cd fiable
bash install.sh
export HF_TOKEN=...
fiable download
fiable quantize
fiable evaluate --limit 50
fiable store
```

`install.sh` installs cmake/build tools, cuBLAS headers (`CUDA::cublas`) when CUDA is present, and the CLI. Quantize uses `/opt/llama.cpp` binaries when they already exist; set `FIABLE_BUILD_LLAMA=1` to compile from `store/llama.cpp` instead.

Default `fiable quantize` / `fiable evaluate` compare **Q4_K_M**, **GPTQ_4**, and **EVOPRESS_4** against FP16. Other GGUF types (`Q8_0`, `Q6_K`, …) remain available with `--types`. GPTQ needs `pip install fiable[gptq]` and CUDA torch. EvoPress builds extra uniform GGUFs as a search database; those files are not in the default eval table.

## Quantization methods

| Method | One-line meaning |
| --- | --- |
| **FP16** | 16-bit floating-point weights; essentially the unquantized baseline. |
| **Q8_0** | 8-bit symmetric block quantization using one scale per block; very low quantization error. |
| **Q6_K** | 6-bit K-quant using hierarchical blocks/superblocks and per-block scaling for better efficiency. |
| **Q5_K_M** | ~5-bit K-quant with **mixed precision**, using higher precision for selected sensitive tensors. |
| **Q4_K_M** | ~4-bit K-quant with **mixed precision**; a popular balance between model size and quality. |
| **GPTQ_4** | Calibrated 4-bit GPTQ (group 128, act-order) on WikiText-2 **train**; not a llama.cpp `llama-quantize` type. |
| **EVOPRESS_4** | EvoPress mixed-precision: evolutionary per-block mix of `{Q2_K…Q6_K}` targeting ~4-bit average. |
| **Q3_K_M** | ~3-bit K-quant with **mixed precision**; more aggressive compression with noticeably higher quantization error. |
| **Q2_K** | ~2-bit K-quant with very aggressive compression; high memory savings but substantially more quality loss. |

### Even shorter version

| Method | Meaning |
| --- | --- |
| **FP16** | Full 16-bit precision |
| **Q8_0** | 8-bit block quantization |
| **Q6_K** | 6-bit hierarchical K-quantization |
| **Q5_K_M** | 5-bit K-quant + mixed tensor precision |
| **Q4_K_M** | 4-bit K-quant + mixed tensor precision |
| **GPTQ_4** | 4-bit GPTQ (g128, calibrated) |
| **EVOPRESS_4** | Mixed GGUF types, ~4-bit EvoPress |
| **Q3_K_M** | 3-bit K-quant + mixed tensor precision |
| **Q2_K** | 2-bit K-quantization |

**Key:** `K` = K-quantization family; `M` = mixed tensor types, not simply “medium quality.” Default comparison is **Q4_K_M** (uniform K-quant recipe) vs **GPTQ_4** (calibrated 4-bit, WikiText-2 train) vs **EVOPRESS_4** (evolutionary per-block mix targeting ~4-bit). GPTQ eval PPL/KL/lm-eval use a dequantized F16 GGUF of reconstructed weights; `size_gb` / bits come from the packed checkpoint; prefill/decode are omitted. EvoPress stitches a mixed K-quant GGUF, so PPL **and** llama-bench are comparable to Q4_K_M.

`W4A16` on the CLI still means **Q4_K_M**, not GPTQ or EvoPress.

## Evaluation report fields

`fiable evaluate` writes `report/evaluation.json`. The `summary` array is one row per GGUF. Relative fields (`compression_ratio`, `delta_*`, `speedup`, …) compare that row to the family’s **baseline** (FP16/BF16 if present, else Q8_0, else the largest file).

### Identity

| Field | Meaning |
| --- | --- |
| `model` | Model family id from the GGUF name (`llama-3.1-8b-instruct`). |
| `quant` | GGUF weight scheme of this file (`FP16`, `Q4_K_M`, …). |
| `baseline` | Scheme used as the uncompressed reference for this family. |

### Size

| Field | Meaning |
| --- | --- |
| `size_gb` | On-disk GGUF size in GiB. |
| `compression_ratio` | `baseline_size / size_gb`. Baseline is `1.0`; `5.0` means 5× smaller. |
| `size_reduction` | Fraction of baseline size removed: `1 - size / baseline_size`. Baseline is `0`; `0.80` is an 80% smaller file. |
| `bits_per_weight` | `8 × file_bytes / n_params`. FP16 is ~16; K-quants sit a bit above their nominal bit width because of scales and super-blocks. |

### Quality (WikiText-2 via `llama-perplexity`)

| Field | Meaning |
| --- | --- |
| `perplexity` | Next-token perplexity at context **512**. Lower is better. |
| `delta_ppl` | Relative PPL change vs baseline: `ppl / ppl_baseline - 1`. Baseline is `0`; `0.39` is 39% worse. |
| `perplexity_long` | Same metric at context **2048** (`LONG_CONTEXT_SIZE`). |
| `kl_divergence` | Mean KL(baseline ‖ quant) on WikiText logits (`--kl-divergence`). `0` on the baseline; higher means the quantized next-token distribution drifted more. |
| `top1_match` | Fraction of tokens where the quant and baseline share the same top-1 token (`Same top p` / 100). Baseline is `1.0`. |

### Speed / memory (`llama-bench`: prompt 512, generate 128, 3 reps)

| Field | Meaning |
| --- | --- |
| `prefill_tok_s` | Prompt-processing (pp) throughput, tokens/s. |
| `decode_tok_s` | Token-generation (tg) throughput, tokens/s. |
| `latency_ms` | Decode latency, `1000 / decode_tok_s` (ms per token). |
| `speedup` | Decode speed vs baseline: `latency_baseline / latency`. Baseline is `1.0`. |
| `prefill_speedup` | Prefill speed vs baseline from estimated TTFT (`n_prompt / prefill_tok_s`). Baseline is `1.0`. Values `< 1` mean slower prefill than FP16 (common for low-bit kernels). |
| `vram_gb` | Peak GPU memory during the bench (NVML sampler, else llama-bench’s reported memory). |

### Task accuracy (lm-eval; task name is the suffix)

| Field | Meaning |
| --- | --- |
| `acc_<task>` | Score on that task (`acc_gsm8k`, `acc_mmlu`, `acc_humaneval`, …). With `--limit 50` this is a short sample, not a full run. |
| `delta_acc_<task>` | Absolute drop vs baseline: `acc_baseline - acc`. Baseline is `0`; **positive** means the quant scored worse. |
| `acc_retention_<task>` | `acc / acc_baseline`. Baseline is `1.0`; `0.71` kept 71% of baseline accuracy. |
| `efficiency_<task>` | Harmonic mean of `acc_retention_<task>` and `size_reduction`. High when the file is much smaller **and** accuracy is mostly kept. Omitted on the baseline (`size_reduction` is 0). |

On an FP16 baseline row: `compression_ratio = 1`, `delta_ppl = 0`, `kl_divergence = 0`, `top1_match = 1`, `speedup = 1`, `acc_retention_gsm8k = 1`.

## Charts (`fiable plot`)

PNG + CSV files land in `report/`. Research plots (no extra eval):

| Chart | File | Question |
| --- | --- | --- |
| Master comparison | `comparison.png` | All models × measures × GGUF methods in one table |
| Relative PPL vs bits | `delta_ppl_vs_bits.png` | Quality loss as bit-width drops (comparable across models) |
| Perplexity drop vs FP16 | `perplexity_drop.png` | Same ΔPPL by GGUF method (not by bits) |
| KL and top-1 vs bits | `kl_top1_vs_bits.png` | Next-token distribution drift |
| Prefill vs decode | `prefill_vs_decode.png` | Decode can speed up while prefill slows vs FP16 |
| Short vs long PPL | `perplexity_short_vs_long.png` | Whether error grows from ctx 512 to 2048 |
| Compression–quality Pareto | `efficiency_pareto.png` | Which quants sit on the size vs PPL-retention front |

Accuracy vs size / throughput / drop use lm-eval scores. **With `--limit 50` those three charts are exploratory** (high binomial noise); they are labeled as such and should not rank quants until you rerun without `--limit` (and add `mmlu` / `humaneval` if needed).


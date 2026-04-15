# LLM VRAM/RAM Calculator

Terminal UI for estimating whether a GGUF LLM will fit in your available `VRAM + RAM`.

The calculator takes a Hugging Face GGUF repo, resolves the underlying base model, reads that model's `config.json`, enumerates available GGUF quantizations, and then estimates:

- model weight memory
- KV-cache / context memory
- VRAM usage
- RAM spillover

## Run

```bash
uv run llm_calculator.py --vram 32 --ram 64
```

Arguments:

- `--vram`: available GPU memory in GB
- `--ram`: available system memory in GB

Inside the TUI:

- `j` / `k`: move down / up
- `h` / `l`: switch panes
- arrow keys: move / switch panes
- `Tab`: switch panes
- `Ctrl+C`: exit

## Input Resolution Logic

Given a GGUF repo such as `unsloth/Qwen3.5-4B-GGUF`, the app resolves the base model in this order:

1. Read the model card metadata with `HfApi().model_info(...)`.
2. Follow `card_data.base_model` recursively until there is no deeper base model.
3. Use the final resolved base model for `config.json`.
4. Use the original GGUF repo for the list of `.gguf` files.

If `base_model` metadata is missing, the app falls back to heuristics based on the repo name. For example, `Qwen...-GGUF` falls back to `Qwen/...`.

## Config Parsing Logic

The calculator reads the base model's `config.json` and prefers nested `text_config` when present. This matters for multimodal and nested configs such as Qwen 3.x models.

The main values it extracts are:

- `num_hidden_layers`
- `num_key_value_heads`
- `head_dim`
- `max_position_embeddings`
- parameter count when explicitly available

Fallback rules:

- KV heads: `num_key_value_heads`, then `num_attention_heads`, then `n_head`
- Head dim: `head_dim`, then `hidden_size / heads`
- Max context: max of known context keys such as `max_position_embeddings`, `max_sequence_length`, `model_max_length`, `seq_len`, plus `rope_scaling` metadata when present
- Parameter count:
  1. `num_parameters`
  2. `parameter_count`
  3. parse a marketed size from identifiers like `Qwen3.5-27B`
  4. rough architecture estimate

## Quantization Logic

The app lists `.gguf` files in the GGUF repo and infers bits-per-weight from the filename.

Examples:

- `Q8_0` -> `8`
- `Q6_K` -> `6`
- `Q4_K_M` -> `4`
- `F16` / `FP16` -> `16`
- `F32` / `FP32` -> `32`

The TUI sorts quantizations from highest precision to lowest precision.

## Math

The calculator currently uses two main formulas.

### 1. Model Weight Memory

```text
model_size_gb = (parameters_in_billions * bits_per_weight) / 8
```

Examples:

- `27B` at `Q4` -> `27 * 4 / 8 = 13.5 GB`
- `27B` at `Q8` -> `27 * 8 / 8 = 27 GB`

This is a simplified weight-size estimate. It intentionally ignores GGUF container overhead and format-specific metadata.

### 2. KV Cache / Context Memory

```text
context_size_gb = (2 * layers * kv_heads * head_dim * context_length) / 1e9
```

Where:

- `2` = keys + values
- `layers` = transformer block count
- `kv_heads` = `num_key_value_heads`
- `head_dim` = per-head width
- `context_length` = selected token window

Example:

```text
layers = 64
kv_heads = 4
head_dim = 256
context = 131072

context_size_gb
= (2 * 64 * 4 * 256 * 131072) / 1e9
= 17.18 GB
```

This is the value the TUI labels as `Context Size`.

## Memory Placement Logic

After computing:

```text
total_needed = model_size_gb + context_size_gb
```

the calculator assumes a simple placement rule:

1. Fill VRAM first.
2. Spill the remainder into RAM.

Implemented as:

```text
vram_used = min(total_needed, vram_total)
ram_used = max(0, total_needed - vram_used)
```

So the tool is answering:

"If this total memory footprint is loaded with VRAM filled first, how much RAM is left to cover the overflow?"

## Context Presets

The TUI offers these presets:

- `4K`
- `8K`
- `16K`
- `32K`
- `64K`
- `128K`
- `256K`

They are capped at the model's detected max context length.

## What The Tool Assumes

This is an intentionally simple estimator, not a runtime profiler.

Assumptions:

- weights are approximated only by `params * bits / 8`
- KV cache uses `num_key_value_heads`
- VRAM fills before RAM
- no explicit allocator overhead is modeled
- no batching effects are modeled
- no speculative decoding, MoE routing, activation memory, or runtime-specific buffers are modeled
- no distinction is made between inference engines such as `llama.cpp`, `vLLM`, `transformers`, etc.

## Why Numbers May Differ From Real Runtime

Actual memory use can be higher or lower depending on:

- GGUF metadata overhead
- runtime buffers and scratch space
- fused kernels and allocator fragmentation
- partial GPU offload settings
- batch size / parallel sequences
- MoE expert activation patterns
- vision/audio tower memory for multimodal models

So the calculator is best treated as a fast first-pass fit estimate.

## Current Scope

The implementation is optimized for the repo flow:

1. enter a GGUF Hugging Face repo
2. resolve its base model
3. estimate fit across available quantizations and context sizes

It is not yet a full accuracy model for every architecture or runtime, but the parsing now correctly handles common modern config patterns such as:

- nested `text_config`
- large-context rope metadata
- marketed parameter counts from model identifiers
- grouped-query attention via `num_key_value_heads`

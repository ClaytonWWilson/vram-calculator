# LLM VRAM/RAM Calculator

This repo now contains two interfaces for the same estimator:

- `cli/`: the original Python terminal UI
- `web/`: a SvelteKit web app with the same Hugging Face lookup flow and memory math

Both versions:

- accept a Hugging Face GGUF repo
- resolve the underlying base model
- read `config.json`
- enumerate available GGUF quantizations
- estimate model weight memory, KV-cache/context memory, VRAM usage, and RAM spillover

## Web App

```bash
cd web
bun install
bun run dev
```

The page lets you:

- enter `VRAM` and `RAM` capacity at the top
- load a GGUF repo from Hugging Face
- select a context preset and quantization
- view model size, context size, total needed, VRAM used, RAM used, and fit/shortfall status

## CLI

```bash
cd cli
uv run llm_calculator.py --vram 32 --ram 64
```

The original CLI usage and implementation notes now live in [cli/README.md](/home/clayton/Documents/projects/vram-calculator/cli/README.md).

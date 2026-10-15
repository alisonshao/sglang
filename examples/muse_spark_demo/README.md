# DemoLM-8B — SGLang Cookbook Demo

Screenshot-to-page demonstration with **synthetic data only**.

## What this is

A self-contained static page that recreates the layout in the two Demo screenshots:

- `cookbook.html` — Cookbook demo page for fictional model **DemoLM-8B**
- No backend, no build step, no external dependencies (embedded CSS + inline SVG)

Prominent labels on the page state `DUMMY DATA` / `Fictional model. Synthetic metrics. Commands have not been executed.` The commands are reference text only.

## How to open (local preview)

```bash
# from repo root
open examples/muse_spark_demo/cookbook.html
# or
python3 -m http.server 8000 --directory examples/muse_spark_demo
# then visit http://127.0.0.1:8000/cookbook.html
```

Any static file server works — the page is a single HTML file.

## Synthetic data notice

All performance numbers are **synthetic / dummy** and have not been measured:

- 3 scenarios × 4 concurrency levels (1 / 8 / 32 / 128)
- Scenarios: `short-chat 512/128`, `long-input 8192/256`, `long-output 1024/2048`
- Table metrics: Total tok/s/GPU, Output tok/s, TTFT P50 (ms), TPOT P50 (ms)
- Charts: Total tok/s/GPU, Output tok/s, TTFT P50, TPOT P50 versus concurrency (inline SVG, no chart library)

Do not use for hardware sizing, deployment decisions, or model selection. Do not download model weights or run GPU inference — the checkpoint path `/models/demo/DemoLM-8B` is fictional.

## Verification

Tables and shell panels were transcribed from the screenshots and rendering was checked locally (responsive layout, code blocks, inline SVG charts). The page should match the screenshots' structure, labels, and numeric values.

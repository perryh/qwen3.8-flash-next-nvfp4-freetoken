# Qwen3.8-Flash-Next-NVFP4 on FreeToken (Docker)

Serves the NVFP4-quantized [Qwen3.8-Flash-Next](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
(125B-A6B hybrid MoE, NVFP4 W4A4 on routed experts) behind
[FreeToken](https://github.com/FlashML-org/FreeToken) — an edge-native MoE
serving engine with CPU↔GPU expert offload — on a single RTX 5090 (32GB) +
Ryzen 5950X + 128GB DDR4.

OpenAI + Anthropic compatible APIs on port 1919.

## Requirements

- NVIDIA RTX 50-series (sm_120) + driver r580+ (CUDA 13)
- Docker + Compose v2; SELinux-enforcing hosts supported (`:ro,Z` volume)
- Model checkpoint at `~/.freetoken/models/Qwen3.8-Flash-Next-NVFP4` (NVFP4 safetensors, converted with NVIDIA Model Optimizer)

## Run

```bash
./run.sh          # writes .env, compose up -d --build, waits for health
./run.sh test     # /health + /v1/models + a live chat completion
./run.sh logs     # follow logs
./run.sh stop
```

Server: `http://localhost:1919` — OpenAI (`/v1/chat/completions`,
`/v1/models`, `/v1/responses`) and Anthropic (`/v1/messages`) APIs.

First start JIT-compiles CUDA kernels — give it several minutes. Model load
after image warm: ~2 minutes to ready (weights 29.3 GiB of the 32 GiB card
with `--memory-ratio 1`).

### Knobs (env / `.env`)

| var | default | meaning |
|---|---|---|
| `PORT` | `1919` | host port |
| `MODEL_DIR` | `~/.freetoken/models/Qwen3.8-Flash-Next-NVFP4` | host path to the checkpoint |
| `EXTRA_ARGS` | `--memory-ratio 1` | extra `ft serve` flags |
| `GPU_DEVICE_ID` | `0` | GPU pin |

Useful `ft serve` flags (pass via `EXTRA_ARGS`):
- `--num-tokens N` — raise KV capacity (default auto-sized; ~8.2K tokens with
  memory-ratio 1 on the 32GB card because the GPU expert cache takes priority)
- `--moe-cache-rate 0.5` — shrink GPU expert cache to free VRAM for KV
- `--quant-backend moe.nvfp4=b12x` — NVFP4 kernel selection (marlin/b12x/triton)
- `ft bench bw` inside the container calibrates the auto MoE strategy

## Benchmark

`bench_final.py` — pure-decode, prefill (4K/8K), and long-context generation.
Measured 2026-09-09 on this repo's defaults (`--memory-ratio 1`):

| metric | FreeToken (NVFP4) | ik_llama MTP (MXFP4) * |
|---|---|---|
| decode (prose) | **~48 tok/s** | ~21 tok/s |
| decode w/ long ctx (8K) | ~53 tok/s | ~21 tok/s |
| prefill @ 4.6K | 1,900 tok/s | ~300 tok/s |
| prefill @ 8.1K | **3,300 tok/s** | ~300 tok/s |
| KV context budget | ~8.2K tokens (default) | 262,144 tokens |

\* ik_llama numbers from its repo README, measured on the same machine
(5950X dual-channel DDR4, q8 KV, spec decode ngram+MTP).

Notes:
- FreeToken wins raw speed ~2x on decode and ~6-10x on prefill: the NVFP4
  experts mostly live in a 19.5GB GPU cache (LRU, 5,925 slots cached at
  capture time), so far fewer expert reads stream from DDR4 than ik_llama's
  offload-everything layout.
- The tradeoff is context: FreeToken's auto-sizing left only ~8.2K KV tokens.
  Raise it with `--num-tokens` + `--moe-cache-size` (see below) — every KV token
  costs 25,344 bytes, and it comes directly out of the expert cache.
- Thinking mode is on by default (reasoning_content separated); 5950X Zen 3
  needs no special flags here — FreeToken handles kernel selection.

## Files

- `Dockerfile` — CUDA 13 devel + uv + FreeToken source install (`[accel]`;
  git + python3.12-dev required: source build compiles a torch C++ extension)
- `docker-compose.yml` — port, `:ro,Z` model volume, GPU reservation, restart policy
- `entrypoint.sh` — `ft serve` args from env knobs (defaults to `--memory-ratio 1`)
- `run.sh` — one-command start/test/logs/stop wrapper
- `bench_final.py` — decode/prefill/long-ctx benchmark (results above)


## Full-context mode (verified working)

`EXTRA_ARGS=--num-tokens 262144 --moe-cache-size 4300` (use 3600 if it OOMs
under load) serves the **full 262,144-token window** on the 32GB card:
KV 6.19GB + mamba 2.64GB + 30.9GB total VRAM. Trade: expert cache drops
5,925 -> ~4,300, decode falls 48 -> ~22 tok/s; prefill stays fast
(~1,600 tok/s @ 14.5K measured). Boot smoke passes at 4,500 but the backend
crashes ~100-160MB short under real load — give it margin.

Default (no flags beyond `--num-tokens 16384`) = fast mode: 16K ctx, ~48 tok/s.

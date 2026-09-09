# Qwen3.8-Flash-Next-NVFP4 on FreeToken (Docker)

Serves the NVFP4-quantized [Qwen3.8-Flash-Next](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
(125B-A6B hybrid MoE, NVFP4 W4A4 on routed experts) behind
[FreeToken](https://github.com/FlashML-org/FreeToken) — an edge-native MoE
serving engine with CPU↔GPU expert offload — on a single RTX 5090 (32GB) +
Ryzen 5950X + 128GB DDR4.

OpenAI + Anthropic compatible APIs on port 1919.

## Two verified profiles

The 32GB card forces a trade between **decode speed** (GPU expert cache) and
**context** (KV pool). Both profiles below are verified stable on this card.

| | Fast mode (default) | Full-context mode |
|---|---|---|
| `EXTRA_ARGS` | `--num-tokens 16384` | `--num-tokens 262144 --moe-cache-size 3600` |
| Context window | 16,384 tokens | **262,144 tokens** (model max) |
| Decode (prose, warm) | **~48 tok/s** | ~22 tok/s |
| Prefill | ~2-3K tok/s | ~1,600 tok/s @ 14.5K |
| GPU expert cache | 5,925 slots | 3,600 slots |
| VRAM | ~29.5 / 32.6 GB | ~30.9 / 32.6 GB |

Switch by editing the `EXTRA_ARGS` default in `docker-compose.yml`, then
`docker compose up -d --force-recreate`. Long-context workloads can also go to
our ik_llama server on :9292 (full 262K at ~21 tok/s decode).

**Critical: do NOT add `--memory-ratio 1`.** The engine default (0.9) budgets
headroom for the 2.64GB mamba linear-state pool; ratio 1.0 packs weights+experts
to the brim and the backend OOMs at startup (`linear_state_pool` alloc). All
memory lessons learned on this card:

- KV costs **25,344 bytes/token**; each cached expert costs 2.77MB. Both come
  out of the same pool after ~26GB of weights.
- `--num-tokens` must be a multiple of the page size (64).
- Boot smoke passing ≠ stable: at expert-cache 4,300-4,500 with 262K KV the
  backend crashes ~100-160MB short **under real load** (runtime workspace
  growth). Verified-stable margin starts at ~3,600 slots. If you push the
  cache size up, stress-test with two decode passes + a 14K-token prompt.
- Small `max_tokens` with thinking mode (default ON) yields empty `content`
  (budget eaten by `reasoning_content`). Give ≥150 tokens or disable thinking.

## Run

```bash
./run.sh          # writes .env, compose up -d --build, waits for health
./run.sh test     # /health + /v1/models + a live chat completion
./run.sh logs     # follow logs
./run.sh stop
```

Server: `http://localhost:1919` — OpenAI (`/v1/chat/completions`,
`/v1/models`, `/v1/responses`) and Anthropic (`/v1/messages`) APIs.

First start JIT-compiles CUDA kernels — give it several minutes. Warm image
load: ~2.5 min to READY (weights ~26GB, then expert banks + pools).

### Knobs (env / `.env`)

| var | default | meaning |
|---|---|---|
| `PORT` | `1919` | host port |
| `MODEL_DIR` | `~/.freetoken/models/Qwen3.8-Flash-Next-NVFP4` | host path to the checkpoint |
| `EXTRA_ARGS` | `--num-tokens 16384` | extra `ft serve` flags |
| `GPU_DEVICE_ID` | `0` | GPU pin |

Useful `ft serve` flags (pass via `EXTRA_ARGS`):
- `--num-tokens N` — KV capacity in tokens (multiple of 64)
- `--moe-cache-size N` — absolute GPU expert-cache cap in slots
- `--moe-cache-rate 0.5` — relative cap (prefer `--moe-cache-size` for exact VRAM math)
- `--quant-backend moe.nvfp4=b12x` — NVFP4 kernel selection (marlin/b12x/triton)
- `ft bench bw` inside the container calibrates the auto MoE strategy
- Do not set `--memory-ratio`; the 0.9 default is required headroom (see above)

## Benchmark

Measured 2026-09-09, same RTX 5090 + 5950X box:

| metric | FreeToken fast (16K) | FreeToken full (262K) | ik_llama MTP * |
|---|---|---|---|
| decode (prose, warm) | **~48 tok/s** | ~22 tok/s | ~21 tok/s |
| prefill @ 4.6K | 1,900 tok/s | — | ~300 tok/s |
| prefill @ 8.1K | **3,300 tok/s** | — | ~300 tok/s |
| prefill @ 14.5K | — | ~1,600 tok/s | — |
| context budget | 16,384 | **262,144** | 262,144 |

\* ik_llama numbers from its repo README (q8 KV, spec decode ngram+MTP).

Notes:
- In fast mode the NVFP4 experts mostly live in the GPU cache (LRU), so far
  fewer expert reads stream from DDR4 than ik_llama's offload-everything
  layout — hence ~2x decode and ~6-10x prefill.
- Full-context mode halves decode (smaller expert cache → more DDR4 fetches)
  but prefill stays fast and the window matches ik_llama's.
- Cache geometry is introspectable at `GET /v1/cache/status` (pages, expert
  slots, per-token/per-expert byte costs).

## Files

- `Dockerfile` — CUDA 13 devel + uv + FreeToken source install (`[accel]`;
  git + python3.12-dev required: source build compiles a torch C++ extension)
- `docker-compose.yml` — port, `:ro,Z` model volume, GPU reservation, restart
  policy; `EXTRA_ARGS` default selects the profile
- `entrypoint.sh` — `ft serve` args from env knobs
- `run.sh` — one-command start/test/logs/stop wrapper
- `bench.py` — decode/prefill/long-ctx benchmark (wall-clock based; FreeToken's
  usage block does not expose per-phase timings)

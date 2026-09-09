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

First start JIT-compiles CUDA kernels and may convert the checkpoint to
FreeToken's FTW fast-load format — give it several minutes.

### Knobs (env / `.env`)

| var | default | meaning |
|---|---|---|
| `PORT` | `1919` | host port |
| `MODEL_DIR` | `~/.freetoken/models/Qwen3.8-Flash-Next-NVFP4` | host path to the checkpoint |
| `EXTRA_ARGS` | *(empty)* | extra `ft serve` flags, e.g. `"--moe-strategy hybrid --moe-cache-rate 0.6"` |
| `GPU_DEVICE_ID` | `0` | GPU pin |

Useful `ft serve` flags (pass via `EXTRA_ARGS`):
- `--moe-strategy {auto,fused,offload,cpu,hybrid}` — expert placement (auto = offload, or hybrid with a bandwidth profile)
- `--moe-cache-rate 0.6` — GPU expert-cache fraction
- `--quant-backend moe.nvfp4=b12x` — NVFP4 kernel selection (marlin/b12x/triton)
- `--max-running-requests N`, `--max-seq-len-override N`
- `ft bench bw` inside the container calibrates the auto strategy (writes a per-GPU profile)

## Notes

- FreeToken resolves dtype/backends/cache sizes automatically from the
  checkpoint + GPU; the NVFP4 experts are detected and an appropriate kernel
  backend is picked. First-use kernel JIT needs the CUDA 13 toolkit — included
  in the build image.
- The container runs `ft serve` from a source checkout of
  FlashML-org/FreeToken (pinned by the Docker build).
- SELinux: model volume is mounted `:ro,Z` (relabel is metadata-only).

## Files

- `Dockerfile` — CUDA 13 devel + uv + FreeToken source install (`[accel]` extra)
- `docker-compose.yml` — port, `:ro,Z` model volume, GPU reservation, restart policy
- `entrypoint.sh` — assembles `ft serve` args from env knobs
- `run.sh` — one-command start/test/logs/stop wrapper

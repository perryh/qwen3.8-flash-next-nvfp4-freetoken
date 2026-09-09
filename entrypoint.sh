#!/usr/bin/env bash
# FreeToken server for Qwen3.8-Flash-Next-NVFP4.
# Knobs overridable via env / .env (written by run.sh).
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.8-Flash-Next-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-1919}"
EXTRA_ARGS="${EXTRA_ARGS:---memory-ratio 1 --num-tokens 262144}"

ARGS=(serve --model "$MODEL_PATH" --host "$HOST" --port "$PORT")

# Optional flags pass-through
if [ -n "$EXTRA_ARGS" ]; then
  # shellcheck disable=SC2086
  ARGS+=( $EXTRA_ARGS )
fi

exec /app/FreeToken/.venv/bin/ft "${ARGS[@]}"

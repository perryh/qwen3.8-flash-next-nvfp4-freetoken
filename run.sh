#!/usr/bin/env bash
# One-command manager for the FreeToken Qwen3.8 NVFP4 server.
# Usage: ./run.sh [test|logs|stop|restart]   (no args = start)
set -euo pipefail
cd "$(dirname "$0")"

MODEL_DIR_DEFAULT="$HOME/.freetoken/models/Qwen3.8-Flash-Next-NVFP4"
PORT="${PORT:-1919}"

if [ ! -d "${MODEL_DIR:-$MODEL_DIR_DEFAULT}" ] && [ -z "${MODEL_DIR:-}" ]; then
  echo "Model not found at $MODEL_DIR_DEFAULT" >&2
  echo "Set MODEL_DIR=/path/to/model and re-run." >&2
  exit 1
fi

cat > .env <<EOF
MODEL_DIR=${MODEL_DIR:-$MODEL_DIR_DEFAULT}
PORT=$PORT
EOF

case "${1:-up}" in
  up|restart)
    docker compose up -d --build
    echo "Waiting for model to load (first load can take several minutes)..."
    for i in $(seq 1 120); do
      if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "✅ Healthy on http://localhost:$PORT"
        echo "   Next: ./run.sh test"
        exit 0
      fi
      sleep 5
    done
    echo "❌ Not healthy after 10 min — check: ./run.sh logs" >&2
    exit 1
    ;;
  test)
    echo "== /health =="
    curl -s "http://localhost:$PORT/health" && echo
    echo "== /v1/models =="
    curl -s "http://localhost:$PORT/v1/models" | head -c 500; echo
    echo "== chat completion =="
    curl -s "http://localhost:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"Qwen3.8-Flash-Next-NVFP4","messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"max_tokens":50}' \
      | head -c 800; echo
    ;;
  logs)
    docker logs -f freetoken
    ;;
  stop)
    docker compose down
    ;;
  *)
    echo "Usage: $0 [up|test|logs|stop|restart]" >&2
    exit 1
    ;;
esac

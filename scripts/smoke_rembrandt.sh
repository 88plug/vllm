#!/usr/bin/env bash
# One-shot completion against a running serve_rembrandt.sh
set -euo pipefail
PORT="${PORT:-8000}"
curl -sS "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{
    "model": "default",
    "messages": [{"role":"user","content":"Reply with exactly: pong"}],
    "max_tokens": 8,
    "temperature": 0
  }' | tee /tmp/vllm-rembrandt-smoke.json
echo

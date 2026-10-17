#!/usr/bin/env bash
# Keep the CashGuard service and its Cloudflare tunnel alive (restart on crash).
# Usage: bash deploy/run.sh            (logs in .build/server.log, .build/tunnel.log)
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PORT="${CASHGUARD_PORT:-8090}"

serve() {
  while true; do
    echo "$(date '+%F %T') starting server on :$PORT" >> .build/server.log
    .venv/bin/python -m datathon.cashguard serve --port "$PORT" >> .build/server.log 2>&1
    echo "$(date '+%F %T') server exited ($?), restarting in 3s" >> .build/server.log
    sleep 3
  done
}

tunnel() {
  while true; do
    cloudflared tunnel --config deploy/cloudflared-cashguard.yml run cashguard \
      >> .build/tunnel.log 2>&1
    echo "$(date '+%F %T') tunnel exited ($?), restarting in 5s" >> .build/tunnel.log
    sleep 5
  done
}

mkdir -p .build
serve &
tunnel &
trap 'kill 0' INT TERM
echo "CashGuard: http://127.0.0.1:$PORT  (public: https://cashguard.zerde.site)"
wait

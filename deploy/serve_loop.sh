#!/usr/bin/env bash
# Run the service and restart it if it ever exits.
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while true; do
  .venv/bin/python -m datathon.cashguard serve --port "${CASHGUARD_PORT:-8090}" >> .build/server.log 2>&1
  echo "$(date '+%F %T') server exited, restarting" >> .build/server.log
  sleep 3
done

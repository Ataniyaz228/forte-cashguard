#!/usr/bin/env bash
# Run the service and restart it if it ever exits. Stopping this script stops the server too
# (otherwise an orphaned server keeps the models on the GPU and the next start runs out of memory).
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
child=0
trap 'kill "$child" 2>/dev/null; wait "$child" 2>/dev/null; exit 0' INT TERM HUP
while true; do
  .venv/bin/python -m datathon.cashguard serve --port "${CASHGUARD_PORT:-8090}" >> .build/server.log 2>&1 &
  child=$!
  wait "$child"
  echo "$(date '+%F %T') server exited, restarting" >> .build/server.log
  sleep 3
done

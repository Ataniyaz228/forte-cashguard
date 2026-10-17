#!/usr/bin/env bash
# Restart the CashGuard service on the RunPod server (detached from the SSH session).
cd /workspace/datathon
pkill -f "deploy/serve_loop.sh" ; pkill -f "datathon.cashguard serve"; sleep 3
setsid bash -c "source deploy/server_env.sh && exec bash deploy/serve_loop.sh" \
  > /dev/null 2>&1 < /dev/null &
for i in $(seq 1 90); do curl -s localhost:8000/health | grep -q ok && break; sleep 2; done
curl -s localhost:8000/health; echo

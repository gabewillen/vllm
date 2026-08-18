#!/bin/bash
# usage: up.sh <yaml> <log>  -> starts server detached, waits for /health on 8013
S=$(cd "$(dirname "$0")" && pwd)
setsid nohup $S/launch.sh "$1" "$2" > /dev/null 2>&1 &
for i in $(seq 1 90); do
  sleep 10
  if curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8013/health 2>/dev/null | grep -q 200; then echo "up ~$((i*10))s"; exit 0; fi
  if grep -q "Engine core initialization failed\|WorkerProc initialization failed\|CUDA out of memory\|Application startup failed" "$2" 2>/dev/null; then echo "startup error:"; grep -m1 -B2 -A12 "Error" "$2" | tail -15; exit 1; fi
done
echo timeout; exit 1

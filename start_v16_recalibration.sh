#!/usr/bin/env bash
set -euo pipefail

PY=".venv/bin/python"
SHARDS="/mnt/d/ChessData/lichess_train_shards"
VALIDATION="training/data/samples/lichess_validation_250k.csv"
INIT="training/models/v1_sf_long_1800m/checkpoint_1.4b.pt"
RUN_DIR="training/models/v1_linear_huber_100m"
LOG="v1_linear_huber_100m.log"

for path in "$PY" "$VALIDATION" "$INIT"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing $path"
    exit 1
  fi
done

if [[ ! -d "$SHARDS" ]]; then
  echo "ERROR: missing training shards: $SHARDS"
  exit 1
fi

mkdir -p "$RUN_DIR"

nohup "$PY" -m training.train_v1_linear_recalibrate \
  --shards "$SHARDS" \
  --validation "$VALIDATION" \
  --init "$INIT" \
  --run-dir "$RUN_DIR" \
  --additional-positions 100m \
  --checkpoint-every 25m \
  --log-every 5m \
  --start-lr 5e-5 \
  --end-lr-factor 0.2 \
  --cp-clip 2000 \
  --huber-beta-cp 200 \
  --device auto \
  > "$LOG" 2>&1 &

PID=$!
echo "$PID" > v1_linear_huber_100m.pid

echo "Started linear-Huber recalibration"
echo "PID: $PID"
echo "Log: $LOG"
echo
echo "Watch:"
echo "  tail -f $LOG"
echo
echo "Check process:"
echo "  ps -p $PID -o pid,etime,cmd"

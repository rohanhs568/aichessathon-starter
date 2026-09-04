#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-.venv-gpu/bin/python}"
TARGET_GLOBAL="${TARGET_GLOBAL:-1800000000}"
INIT="${INIT:-training/models/v1_natural_600m/final.pt}"
RUN_DIR="${RUN_DIR:-training/models/v1_sf_long_1800m}"
SHARDS="${SHARDS:-/mnt/d/ChessData/lichess_train_shards}"
VALIDATION="${VALIDATION:-training/data/samples/lichess_validation_250k.csv}"
LC0_ROOT="${LC0_ROOT:-/mnt/d/ChessData/lc0/test79_random}"

mkdir -p "$RUN_DIR"

echo "============================================================"
echo "1. FULL DATA AUDIT"
echo "============================================================"

"$PY" -m training.audit_all_data \
  --stockfish-shards "$SHARDS" \
  --validation "$VALIDATION" \
  --lc0-root "$LC0_ROOT" \
  --full-zstd-test \
  2>&1 | tee "$RUN_DIR/data_audit.log"

echo
echo "============================================================"
echo "2. GPU PREFLIGHT"
echo "============================================================"

DEVICE="cpu"

if bash training/gpu_preflight.sh \
    2>&1 | tee "$RUN_DIR/gpu_preflight_wrapper.log"; then
  DEVICE="cuda"
  echo "GPU preflight passed. Long run will start on CUDA."
else
  DEVICE="cpu"
  echo "GPU preflight failed. Long run will use CPU."
fi

read_positions () {
  local checkpoint="$1"
  "$PY" - "$checkpoint" <<'PY'
import sys
import torch

path = sys.argv[1]
ckpt = torch.load(path, map_location="cpu", weights_only=False)
print(int(ckpt.get("positions_seen", 0)))
PY
}

START_GLOBAL="$(read_positions "$INIT")"

if [[ "$START_GLOBAL" -le 0 ]]; then
  echo "Could not determine positions_seen from $INIT"
  exit 4
fi

ADDITIONAL=$((TARGET_GLOBAL - START_GLOBAL))

if [[ "$ADDITIONAL" -le 0 ]]; then
  echo "Target $TARGET_GLOBAL is not above checkpoint $START_GLOBAL"
  exit 5
fi

echo
echo "============================================================"
echo "3. LONG STOCKFISH-ONLY CONTROL"
echo "============================================================"
echo "Initial checkpoint: $INIT"
echo "Global start: $START_GLOBAL"
echo "Global target: $TARGET_GLOBAL"
echo "Additional: $ADDITIONAL"
echo "Device: $DEVICE"
echo "Run dir: $RUN_DIR"

set +e
"$PY" -m training.train_v1_continue_sf \
  --shards "$SHARDS" \
  --validation "$VALIDATION" \
  --init "$INIT" \
  --run-dir "$RUN_DIR" \
  --additional-positions "$ADDITIONAL" \
  --checkpoint-every 100m \
  --log-every 10m \
  --device "$DEVICE" \
  2>&1 | tee "$RUN_DIR/long_run_${DEVICE}.log"
status=${PIPESTATUS[0]}
set -e

if [[ $status -eq 0 ]]; then
  echo
  echo "Long run completed successfully on $DEVICE."
  exit 0
fi

if [[ "$DEVICE" != "cuda" ]]; then
  echo "CPU long run failed with exit status $status."
  exit "$status"
fi

echo
echo "CUDA long run failed with status $status."
echo "Attempting automatic CPU recovery from the newest checkpoint."

LATEST="$("$PY" - "$RUN_DIR" "$INIT" <<'PY'
import sys
from pathlib import Path
import torch

run_dir = Path(sys.argv[1])
fallback = Path(sys.argv[2])

candidates = list(run_dir.glob("checkpoint_*.pt"))
if not candidates:
    print(fallback)
    raise SystemExit

best_path = fallback
best_seen = -1

for path in candidates:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        seen = int(ckpt.get("positions_seen", 0))
    except Exception:
        continue

    if seen > best_seen:
        best_seen = seen
        best_path = path

print(best_path)
PY
)"

RECOVER_START="$(read_positions "$LATEST")"
RECOVER_ADDITIONAL=$((TARGET_GLOBAL - RECOVER_START))

if [[ "$RECOVER_ADDITIONAL" -le 0 ]]; then
  echo "Latest checkpoint already reached target."
  exit 0
fi

echo "Recovery checkpoint: $LATEST"
echo "Recovery global start: $RECOVER_START"
echo "Recovery additional: $RECOVER_ADDITIONAL"

"$PY" -m training.train_v1_continue_sf \
  --shards "$SHARDS" \
  --validation "$VALIDATION" \
  --init "$LATEST" \
  --run-dir "$RUN_DIR" \
  --additional-positions "$RECOVER_ADDITIONAL" \
  --checkpoint-every 100m \
  --log-every 10m \
  --device cpu \
  2>&1 | tee "$RUN_DIR/long_run_cpu_recovery.log"

echo
echo "CPU recovery completed."

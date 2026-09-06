#!/usr/bin/env bash
set -euo pipefail

PY=".venv/bin/python"
BASE="training/models/v1_sf_long_1800m/checkpoint_1.4b.pt"
RUN="training/models/v1_linear_huber_100m"

CHECKPOINTS=("$BASE")

for name in \
  checkpoint_linear_25m.pt \
  checkpoint_linear_50m.pt \
  checkpoint_linear_75m.pt \
  checkpoint_linear_100m.pt
do
  if [[ -f "$RUN/$name" ]]; then
    CHECKPOINTS+=("$RUN/$name")
  fi
done

"$PY" -m training.diagnose_linear_recalibration "${CHECKPOINTS[@]}"

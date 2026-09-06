#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 training/models/v1_linear_huber_100m/checkpoint_linear_50m.pt"
  exit 2
fi

PY=".venv/bin/python"
CKPT="$1"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: missing checkpoint $CKPT"
  exit 1
fi

STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP="candidate_backups/pre_linear_install_$STAMP"
mkdir -p "$BACKUP"
cp weights/v1.npz "$BACKUP/v1.npz"

TMP="weights/v1.linear.tmp.npz"
rm -f "$TMP"

"$PY" training/export_v1_weights.py "$CKPT" "$TMP"
"$PY" -m training.verify_linear_export "$CKPT" "$TMP"

mv "$TMP" weights/v1.npz

echo
echo "Installed linear checkpoint: $CKPT"
echo "Backup: $BACKUP/v1.npz"
echo
echo "Now run:"
echo "  .venv/bin/python -m training.verify_candidate_capture"
echo "  .venv/bin/python -m training.test_candidate_repetition"

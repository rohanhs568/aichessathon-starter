#!/usr/bin/env bash
set -euo pipefail

PY=".venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: .venv/bin/python not found"
  exit 1
fi

echo "[1/5] Capture correctness"
"$PY" -m training.verify_candidate_capture

echo
echo "[2/5] Exact repetition correctness"
"$PY" -m training.test_candidate_repetition

echo
echo "[3/5] Smoke game vs random"
"$PY" -m harness.play --white . --black baselines/random --base-ms 5000 --ply-cap 80

echo
echo "[4/5] Direct A/B vs pre-change V1.4B snapshot"
"$PY" -m harness.arena \
  --opponent baselines/v1_4b_pre_fastasp \
  --games 12 \
  --base-ms 10000

echo
echo "[5/5] Build submission"
"$PY" -m harness.package --out submission_v1_5_fastasp.zip

echo
ls -lh agent.py weights/v1.npz submission_v1_5_fastasp.zip
echo
echo "READY: submission_v1_5_fastasp.zip"

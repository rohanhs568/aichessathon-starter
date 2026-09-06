#!/usr/bin/env bash
set -euo pipefail

if [[ -x ".venv/bin/python" ]]; then
    PY=(".venv/bin/python")
elif [[ -x ".venv-gpu/bin/python" ]]; then
    PY=(".venv-gpu/bin/python")
elif command -v python3 >/dev/null 2>&1; then
    PY=("python3")
elif command -v uv >/dev/null 2>&1; then
    PY=("uv" "run" "python")
else
    echo "ERROR: no usable Python interpreter found"
    exit 127
fi

echo "Using Python: ${PY[*]}"
"${PY[@]}" --version

if [[ ! -f "candidate/agent.py" ]]; then
    echo "ERROR: candidate/agent.py missing. Extract the bundle in repo root first."
    exit 2
fi
if [[ ! -f "training/export_v1_weights.py" ]]; then
    echo "ERROR: training/export_v1_weights.py missing."
    exit 2
fi
if [[ ! -f "training/verify_v1_agent.py" ]]; then
    echo "ERROR: training/verify_v1_agent.py missing."
    exit 2
fi

MODEL_DIR="training/models/v1_sf_long_1800m"
CKPT=""
for candidate in \
    "$MODEL_DIR/checkpoint_1.4b.pt" \
    "$MODEL_DIR/checkpoint_1400m.pt" \
    "$MODEL_DIR/checkpoint_1400M.pt"
do
    if [[ -f "$candidate" ]]; then
        CKPT="$candidate"
        break
    fi
done

if [[ -z "$CKPT" ]]; then
    echo "ERROR: could not find the 1.4b checkpoint. Available checkpoints:"
    ls -lh "$MODEL_DIR"/*.pt 2>/dev/null || true
    exit 3
fi

echo "Checkpoint: $CKPT"

# Require that this really is the deep checkpoint, rather than silently
# deploying the old 600m model under a new filename.
"${PY[@]}" - "$CKPT" <<'PY'
import sys, torch
p=sys.argv[1]
c=torch.load(p, map_location="cpu", weights_only=False)
pos=int(c.get("positions_seen", 0))
print(f"Checkpoint positions_seen: {pos:,}")
if pos < 1_350_000_000:
    raise SystemExit("ERROR: checkpoint is not the expected ~1.4b model")
PY

STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP="candidate_backups/$STAMP"
mkdir -p "$BACKUP" weights

if [[ -f agent.py ]]; then
    cp agent.py "$BACKUP/agent.py"
fi
if [[ -f weights/v1.npz ]]; then
    cp weights/v1.npz "$BACKUP/v1.npz"
fi
printf '%s\n' "$BACKUP" > .candidate_last_backup

echo "Backup: $BACKUP"

echo
echo "[1/7] Installing No-LMR + exact-repetition candidate agent"
cp candidate/agent.py agent.py

echo "[2/7] Exporting 1.4b evaluator"
TMP_WEIGHTS="weights/v1.candidate.tmp.npz"
rm -f "$TMP_WEIGHTS"
"${PY[@]}" training/export_v1_weights.py "$CKPT" "$TMP_WEIGHTS"
mv "$TMP_WEIGHTS" weights/v1.npz

echo "[3/7] PyTorch/export/runtime equivalence + legal-move smoke"
"${PY[@]}" training/verify_v1_agent.py --checkpoint "$CKPT" --positions 250

echo "[4/7] Fundamental capture sanity"
"${PY[@]}" -m training.verify_candidate_capture

echo "[5/7] Exact threefold protocol/history test"
"${PY[@]}" -m training.test_candidate_repetition

echo "[6/7] Short harness smoke vs random"
"${PY[@]}" -m harness.play --white . --black baselines/random --base-ms 5000 --ply-cap 80

echo "[7/7] Building tournament submission"
OUT="submission_v1_4b_nolmr_rep.zip"
rm -f "$OUT"
"${PY[@]}" -m harness.package --out "$OUT"

echo
ls -lh agent.py weights/v1.npz "$OUT"
echo
echo "CANDIDATE READY"
echo "  agent:       V1 FastQ, LMR disabled, exact threefold tracking"
echo "  evaluator:   $CKPT"
echo "  submission:  $OUT"
echo "  rollback:    ./rollback_candidate.sh"
echo
echo "Upload $OUT to the Chessathon site."

#!/usr/bin/env bash
set -euo pipefail

FENS="${FENS:-training/data/test_fens_60.txt}"
CHECKPOINT="${CHECKPOINT:-training/models/v1_natural_600m/final.pt}"
SF="${SF:-/usr/games/stockfish}"
MAX_FENS="${MAX_FENS:-20}"
SF_DEPTH="${SF_DEPTH:-18}"
TIME_LEFT_MS="${TIME_LEFT_MS:-120000}"
OUT="${OUT:-search_lab_results_core20}"

run_py() {
  if [[ -n "${PY:-}" ]]; then
    "$PY" "$@"
  else
    uv run python "$@"
  fi
}

echo "============================================================"
echo "0. CP/tanh metric audit"
echo "============================================================"
run_py -m training.cp_tanh_diagnostics \
  --checkpoint "$CHECKPOINT" \
  --validation training/data/samples/lichess_validation_250k.csv \
  --output-dir cp_tanh_diagnostics

echo
echo "============================================================"
echo "1. Search-lab baseline equivalence"
echo "============================================================"
run_py -m training.verify_searchlab_baseline \
  --original agent.py \
  --lab training/searchlab_agent.py \
  --weights weights/v1.npz \
  --fens "$FENS" \
  --max-fens 10 \
  --depth 5

echo
echo "============================================================"
echo "2. Repetition design tests"
echo "============================================================"
run_py -m training.test_repetition_design \
  --agent-lab training/searchlab_agent.py \
  --weights weights/v1.npz \
  --output repetition_design_test.txt

echo
echo "============================================================"
echo "3. Core paired ablation at fixed nominal depth"
echo "============================================================"
run_py -m training.search_ablation_suite \
  --agent-lab training/searchlab_agent.py \
  --weights weights/v1.npz \
  --fens "$FENS" \
  --max-fens "$MAX_FENS" \
  --preset core \
  --mode fixed-depth \
  --agent-depth 5 \
  --stockfish "$SF" \
  --sf-depth "$SF_DEPTH" \
  --output-dir "${OUT}_fixed"

echo
echo "============================================================"
echo "4. Core paired ablation at tournament clock"
echo "============================================================"
run_py -m training.search_ablation_suite \
  --agent-lab training/searchlab_agent.py \
  --weights weights/v1.npz \
  --fens "$FENS" \
  --max-fens "$MAX_FENS" \
  --preset core \
  --mode timed \
  --time-left-ms "$TIME_LEFT_MS" \
  --stockfish "$SF" \
  --sf-depth "$SF_DEPTH" \
  --output-dir "${OUT}_timed"

echo
echo "DONE"
echo "Read: ${OUT}_fixed/search_ablation_summary.txt"
echo "Read: ${OUT}_timed/search_ablation_summary.txt"

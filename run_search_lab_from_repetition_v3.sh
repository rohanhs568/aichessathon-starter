#!/usr/bin/env bash
set -u
set -o pipefail

echo "============================================================"
echo "2. Repetition design tests (V3)"
echo "============================================================"

# Remove stale bytecode from earlier bundles. Source files are authoritative.
rm -rf training/__pycache__

REP_OK=1
if ! python -m training.test_repetition_design \
    --agent-lab training/searchlab_agent.py \
    --weights weights/v1.npz \
    --output repetition_design_test.txt
then
    REP_OK=0
    echo
    echo "WARNING: repetition diagnostic failed."
    echo "Core null/LMR/futility ablations do not use repetition mode,"
    echo "so the search experiments will continue. Do not integrate"
    echo "repetition into production until the diagnostic passes."
fi

echo
echo "============================================================"
echo "3. Core search ablation: fixed depth 5"
echo "============================================================"
python -m training.search_ablation_suite \
    --agent-lab training/searchlab_agent.py \
    --weights weights/v1.npz \
    --fens training/data/test_fens_60.txt \
    --max-fens 20 \
    --preset core \
    --mode fixed-depth \
    --agent-depth 5 \
    --stockfish /usr/games/stockfish \
    --sf-depth 18 \
    --output-dir search_lab_results_core20_fixed

echo
echo "============================================================"
echo "4. Core search ablation: tournament clock"
echo "============================================================"
python -m training.search_ablation_suite \
    --agent-lab training/searchlab_agent.py \
    --weights weights/v1.npz \
    --fens training/data/test_fens_60.txt \
    --max-fens 20 \
    --preset core \
    --mode timed \
    --time-left-ms 120000 \
    --stockfish /usr/games/stockfish \
    --sf-depth 18 \
    --output-dir search_lab_results_core20_timed

echo
echo "============================================================"
echo "SEARCH LAB V3 COMPLETE"
echo "============================================================"
echo "Repetition test passed: $REP_OK"
echo
echo "Paste these:"
echo "  repetition_design_test.txt"
echo "  search_lab_results_core20_fixed/search_ablation_summary.txt"
echo "  search_lab_results_core20_timed/search_ablation_summary.txt"

#!/usr/bin/env bash
set -u
set -o pipefail

# Resolve a Python interpreter explicitly.
# The user's WSL shell does not provide a bare `python` command unless a venv is active.
if [[ -x ".venv/bin/python" ]]; then
    PY=( ".venv/bin/python" )
elif [[ -x ".venv-gpu/bin/python" ]]; then
    PY=( ".venv-gpu/bin/python" )
elif command -v python3 >/dev/null 2>&1; then
    PY=( "python3" )
elif command -v uv >/dev/null 2>&1; then
    PY=( "uv" "run" "python" )
else
    echo "ERROR: Could not find Python."
    echo "Expected one of:"
    echo "  .venv/bin/python"
    echo "  .venv-gpu/bin/python"
    echo "  python3"
    echo "  uv run python"
    exit 127
fi

echo "Using Python: ${PY[*]}"
"${PY[@]}" --version
echo

echo "============================================================"
echo "2. Repetition design tests (V4 runner)"
echo "============================================================"

# Remove stale bytecode from earlier bundles. Source files are authoritative.
rm -rf training/__pycache__

REP_OK=1
if ! "${PY[@]}" -m training.test_repetition_design \
    --agent-lab training/searchlab_agent.py \
    --weights weights/v1.npz \
    --output repetition_design_test.txt
then
    REP_OK=0
    echo
    echo "WARNING: repetition diagnostic failed."
    echo "Core null/LMR/futility ablations do not use repetition mode,"
    echo "so the search experiments will continue."
fi

echo
echo "============================================================"
echo "3. Core search ablation: fixed depth 5"
echo "============================================================"
"${PY[@]}" -m training.search_ablation_suite \
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
"${PY[@]}" -m training.search_ablation_suite \
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
echo "SEARCH LAB COMPLETE"
echo "============================================================"
echo "Repetition test passed: $REP_OK"
echo
echo "Paste these:"
echo "  repetition_design_test.txt"
echo "  search_lab_results_core20_fixed/search_ablation_summary.txt"
echo "  search_lab_results_core20_timed/search_ablation_summary.txt"

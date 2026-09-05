#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -x ".venv/bin/python" ]]; then
    PY=(".venv/bin/python")
elif [[ -x ".venv-gpu/bin/python" ]]; then
    PY=(".venv-gpu/bin/python")
elif command -v python3 >/dev/null 2>&1; then
    PY=("python3")
elif command -v uv >/dev/null 2>&1; then
    PY=("uv" "run" "python")
else
    echo "ERROR: no Python interpreter found"
    exit 127
fi

if [[ ! -f training/deep_pathology_pipeline.py ]]; then
    echo "ERROR: training/deep_pathology_pipeline.py is missing"
    exit 1
fi

if [[ ! -f training/searchlab_agent.py ]]; then
    echo "ERROR: training/searchlab_agent.py is missing"
    echo "Use the V3 search-lab bundle first."
    exit 1
fi

if [[ ! -f weights/v1.npz ]]; then
    echo "ERROR: weights/v1.npz is missing"
    exit 1
fi

STOCKFISH="/usr/games/stockfish"
if [[ ! -x "$STOCKFISH" ]]; then
    if command -v stockfish >/dev/null 2>&1; then
        STOCKFISH="$(command -v stockfish)"
    else
        echo "ERROR: Stockfish not found at /usr/games/stockfish or PATH"
        exit 1
    fi
fi

mkdir -p deep_pathology_results
LOG="deep_pathology_master.log"
PIDFILE="deep_pathology.pid"

if [[ -f "$PIDFILE" ]]; then
    OLD_PID="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "A deep pathology job is already running with PID $OLD_PID"
        echo "tail -f $LOG"
        exit 0
    fi
fi

echo "Using Python: ${PY[*]}"
"${PY[@]}" --version
echo "Using Stockfish: $STOCKFISH"
echo "Starting an unattended 4.5-hour diagnostic run..."

nohup "${PY[@]}" -u -m training.deep_pathology_pipeline \
    --repo . \
    --agent-lab training/searchlab_agent.py \
    --weights weights/v1.npz \
    --stockfish "$STOCKFISH" \
    --validation training/data/samples/lichess_validation_250k.csv \
    --fen-suite training/data/test_fens_60.txt \
    --regression-csv training/data/search_regression_cases.csv \
    --output-dir deep_pathology_results \
    --budget-hours 4.5 \
    --max-pool 700 \
    --max-pgn-positions 250 \
    --validation-per-phase 100 \
    --screen-agent-depth 5 \
    --screen-agent-cap-s 6 \
    --screen-sf-depth 14 \
    --confirm-sf-depth 18 \
    --screen-candidates 50 \
    --max-pathologies 18 \
    --deep-sf-depth 22 \
    --deep-agent-depth 7 \
    --deep-search-cap-s 20 \
    --time-left-ms 120000 \
    --timed-retest-cases 10 \
    > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PIDFILE"

echo
echo "Started PID $PID"
echo "Log: $LOG"
echo "Results: deep_pathology_results/"
echo
echo "Check progress with:"
echo "  tail -f $LOG"
echo
echo "Check whether it is still running with:"
echo "  ps -p $PID -o pid,etime,%cpu,%mem,cmd"
echo
echo "When you return, show me:"
echo "  cat deep_pathology_results/FINAL_REPORT.txt"
echo "  column -s, -t < deep_pathology_results/06_attribution.csv | less -S"

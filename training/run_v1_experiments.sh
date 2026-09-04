#!/usr/bin/env bash
set -u

# Broad V1 data experiments.
# Architecture and target are fixed. We vary only two high-value data choices:
#   1) natural vs rebalanced training weight profile
#   2) all teacher depths vs minimum depth 22
#
# Four 250m-position runs give a clean 2x2 experiment. On the current machine
# this should fit comfortably inside an unattended ~12 hour window unless the
# depth-filtered runs are much slower than expected.

SHARDS=/mnt/d/ChessData/lichess_train_shards
VALIDATION=training/data/samples/lichess_validation_250k.csv
MAX_POSITIONS=250m
CHECKPOINTS=10m,50m,100m,150m,200m,250m

mkdir -p training/models/v1_experiments

run_experiment () {
    name="$1"
    profile="$2"
    min_depth="$3"

    echo
    echo "============================================================"
    echo "RUNNING $name"
    echo "profile=$profile min_depth=$min_depth"
    echo "============================================================"

    python training/train_v1.py \
        --shards "$SHARDS" \
        --validation "$VALIDATION" \
        --run-dir "training/models/v1_experiments/$name" \
        --hidden 64 \
        --buckets 8 \
        --batch-size 2048 \
        --k-cp 400 \
        --data-profile "$profile" \
        --min-depth "$min_depth" \
        --max-positions "$MAX_POSITIONS" \
        --checkpoints "$CHECKPOINTS" \
        --log-every 10m \
        --learning-rate 1e-3 \
        --end-lr-factor 0.1 \
        2>&1 | tee "training/models/v1_experiments/${name}.log"

    status=${PIPESTATUS[0]}

    if [ "$status" -ne 0 ]; then
        echo "FAILED: $name (exit $status)"
        echo "Stopping the unattended suite rather than continuing with a possibly unhealthy CUDA context."
        exit "$status"
    fi

    echo "COMPLETED: $name"
}

# 2 x 2 factorial on two major data decisions.
run_experiment v1_natural_all_depths      natural    0
run_experiment v1_rebalanced_all_depths   rebalanced 0
run_experiment v1_natural_depth22         natural    22
run_experiment v1_rebalanced_depth22      rebalanced 22

echo
echo "============================================================"
echo "ALL V1 EXPERIMENTS FINISHED"
echo "============================================================"

python training/summarize_v1_experiments.py \
    training/models/v1_experiments

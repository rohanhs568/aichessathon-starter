#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f .candidate_last_backup ]]; then
    echo "ERROR: .candidate_last_backup not found"
    exit 2
fi
BACKUP=$(cat .candidate_last_backup)
if [[ ! -d "$BACKUP" ]]; then
    echo "ERROR: backup directory no longer exists: $BACKUP"
    exit 2
fi

if [[ -f "$BACKUP/agent.py" ]]; then
    cp "$BACKUP/agent.py" agent.py
fi
if [[ -f "$BACKUP/v1.npz" ]]; then
    mkdir -p weights
    cp "$BACKUP/v1.npz" weights/v1.npz
fi

echo "Restored agent.py and weights/v1.npz from $BACKUP"

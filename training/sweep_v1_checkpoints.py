"""Evaluate every saved V1 checkpoint on the permanent held-out validation set.

This is a convergence/reference sweep only. It does not train anything.

Run from repo root, preferably in the background:
    nohup python -m training.sweep_v1_checkpoints \
      --run-dir training/models/v1_natural_600m \
      --validation training/data/samples/lichess_validation_250k.csv \
      > training/models/v1_natural_600m/convergence_sweep.log 2>&1 &

Outputs:
    training/models/v1_natural_600m/convergence_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training import train_v1


def checkpoint_position(path: Path) -> int:
    if path.name == "final.pt":
        return 10**30

    match = re.search(r"checkpoint_([0-9.]+)([kmb]?)\.pt$", path.name.lower())
    if not match:
        return 10**29

    value = float(match.group(1))
    suffix = match.group(2)
    multiplier = {"": 1, "k": 1_000, "m": 1_000_000, "b": 1_000_000_000}[suffix]
    return int(value * multiplier)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    paths = sorted(
        list(args.run_dir.glob("checkpoint_*.pt"))
        + ([args.run_dir / "final.pt"] if (args.run_dir / "final.pt").exists() else []),
        key=checkpoint_position,
    )

    if not paths:
        raise SystemExit(f"No checkpoints found in {args.run_dir}")

    first = torch.load(paths[0], map_location="cpu", weights_only=False)
    first_config = first.get("config", {})
    k_cp = float(first_config.get("k_cp", 400.0))

    print("Loading permanent held-out validation set...", flush=True)
    validation = train_v1.load_validation(args.validation, k_cp)

    device = torch.device("cpu")
    rows: list[dict[str, object]] = []

    for path in paths:
        print(f"\nEvaluating {path.name}", flush=True)

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})

        hidden = int(config.get("hidden", 64))
        buckets = int(config.get("buckets", 8))
        this_k = float(config.get("k_cp", k_cp))

        if this_k != k_cp:
            raise RuntimeError(
                f"k_cp changed across checkpoints: {path.name} has {this_k}, expected {k_cp}"
            )

        model = train_v1.V1Evaluator(hidden, buckets)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()

        metrics = train_v1.evaluate_model(
            model,
            validation,
            device,
            k_cp,
            args.batch_size,
        )

        positions_seen = int(checkpoint.get("positions_seen", 0))

        row = {
            "checkpoint": path.name,
            "positions_seen": positions_seen,
            **metrics,
        }
        rows.append(row)

        print(
            f"positions={positions_seen:,} "
            f"target_mae={metrics['target_mae']:.5f} "
            f"balanced_cp_mae={metrics['balanced_cp_mae']:.2f} "
            f"medium_cp_mae={metrics['medium_cp_mae']:.2f} "
            f"sign={100 * metrics['sign_accuracy']:.2f}% "
            f"mate_sign={100 * metrics['mate_sign_accuracy']:.2f}%",
            flush=True,
        )

    out = args.out or (args.run_dir / "convergence_sweep.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 78)
    print("V1 CHECKPOINT CONVERGENCE")
    print("=" * 78)
    print(
        f"{'positions':>12} {'target':>9} {'bal_cp':>9} "
        f"{'med_cp':>9} {'sign':>8} {'mate':>8}"
    )

    for row in rows:
        print(
            f"{int(row['positions_seen']):>12,} "
            f"{float(row['target_mae']):>9.5f} "
            f"{float(row['balanced_cp_mae']):>9.1f} "
            f"{float(row['medium_cp_mae']):>9.1f} "
            f"{100 * float(row['sign_accuracy']):>7.2f}% "
            f"{100 * float(row['mate_sign_accuracy']):>7.2f}%"
        )

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

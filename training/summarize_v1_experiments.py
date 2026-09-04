import argparse
import json
from pathlib import Path


def final_result(run_dir):
    path = run_dir / "results.jsonl"
    if not path.exists():
        return None

    rows = []
    with path.open() as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        return None

    return max(rows, key=lambda row: row["positions_seen"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    rows = []

    for run_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        result = final_result(run_dir)
        if result is None:
            continue

        config_path = run_dir / "config.json"
        config = {}
        if config_path.exists():
            with config_path.open() as file:
                config = json.load(file)

        rows.append(
            {
                "name": run_dir.name,
                "profile": config.get("data_profile", "?"),
                "depth": config.get("min_depth", "?"),
                "seen": result["positions_seen"],
                "target_mae": result["target_mae"],
                "balanced_cp_mae": result["balanced_cp_mae"],
                "medium_cp_mae": result["medium_cp_mae"],
                "sign_accuracy": result["sign_accuracy"],
                "mate_sign_accuracy": result["mate_sign_accuracy"],
                "within_50_balanced": result["within_50_balanced"],
            }
        )

    if not rows:
        print("No completed checkpoint results found.")
        return

    rows.sort(key=lambda row: row["balanced_cp_mae"])

    print()
    print("V1 EXPERIMENT SUMMARY")
    print("=" * 118)
    print(
        f"{'run':34} {'profile':11} {'depth':>5} {'seen':>12} "
        f"{'target':>8} {'bal MAE':>9} {'med MAE':>9} "
        f"{'sign':>8} {'mate':>8} {'<=50':>8}"
    )
    print("-" * 118)

    for row in rows:
        print(
            f"{row['name']:34} "
            f"{row['profile']:11} "
            f"{row['depth']:>5} "
            f"{row['seen']:>12,} "
            f"{row['target_mae']:>8.4f} "
            f"{row['balanced_cp_mae']:>9.1f} "
            f"{row['medium_cp_mae']:>9.1f} "
            f"{100 * row['sign_accuracy']:>7.2f}% "
            f"{100 * row['mate_sign_accuracy']:>7.2f}% "
            f"{100 * row['within_50_balanced']:>7.2f}%"
        )

    print()
    print(
        "Sorted by balanced-position CP MAE. This is a development metric only; "
        "engine Elo will decide the final model."
    )


if __name__ == "__main__":
    main()

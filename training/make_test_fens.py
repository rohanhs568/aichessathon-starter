"""Sample balanced, varied positions from the permanent validation CSV."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import chess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("training/data/test_fens.txt"))
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--max-abs-cp", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    phase_groups = {
        "opening": [],
        "middlegame": [],
        "endgame": [],
    }

    with args.validation.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            if not row.get("cp_white"):
                continue

            try:
                cp = float(row["cp_white"])
            except ValueError:
                continue

            if abs(cp) > args.max_abs_cp:
                continue

            try:
                board = chess.Board(row["fen"])
            except ValueError:
                continue

            if board.is_game_over(claim_draw=True):
                continue

            pieces = len(board.piece_map())
            if pieces >= 25:
                group = "opening"
            elif pieces >= 13:
                group = "middlegame"
            else:
                group = "endgame"

            phase_groups[group].append(board.fen())

    per_phase = max(1, args.count // 3)
    chosen = []

    for group in ("opening", "middlegame", "endgame"):
        candidates = phase_groups[group]
        rng.shuffle(candidates)
        chosen.extend(candidates[:per_phase])

    remaining = args.count - len(chosen)
    if remaining > 0:
        pool = [
            fen
            for group in phase_groups.values()
            for fen in group
            if fen not in chosen
        ]
        rng.shuffle(pool)
        chosen.extend(pool[:remaining])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(chosen) + "\n", encoding="utf-8")

    print(f"Saved {len(chosen)} FENs to {args.out}")
    for group, candidates in phase_groups.items():
        print(f"{group}: {len(candidates):,} eligible positions")


if __name__ == "__main__":
    main()

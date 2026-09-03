import argparse
import csv
import json
import subprocess
import time
from pathlib import Path


def get_best_evaluation(record):
    """
    Choose the highest-depth evaluation in a Lichess record,
    then take the first PV, which is Stockfish's preferred move.
    """
    evals = record.get("evals", [])

    usable = [
        evaluation
        for evaluation in evals
        if evaluation.get("pvs")
    ]

    if not usable:
        return None

    best = max(
        usable,
        key=lambda evaluation: (
            evaluation.get("depth", -1),
            evaluation.get("knodes", -1),
        ),
    )

    pv = best["pvs"][0]

    if "cp" not in pv and "mate" not in pv:
        return None

    return best, pv


def piece_count_from_fen(fen):
    board_part = fen.split()[0]

    return sum(
        char.isalpha()
        for char in board_part
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--min-depth",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        ["zstd", "-dc", str(args.input.resolve())],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1024 * 1024,
    )

    scanned = 0
    kept = 0
    start = time.time()

    with args.output.open("w", newline="") as output_file:
        writer = csv.writer(output_file)

        writer.writerow([
            "fen",
            "side_to_move",
            "cp_white",
            "mate_white",
            "target_cp_stm",
            "target_mate_stm",
            "depth",
            "knodes",
            "piece_count",
        ])

        try:
            for line in process.stdout:
                scanned += 1

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                fen = record.get("fen")

                if not fen:
                    continue

                result = get_best_evaluation(record)

                if result is None:
                    continue

                evaluation, pv = result

                depth = evaluation.get("depth", -1)

                if depth < args.min_depth:
                    continue

                fields = fen.split()

                if len(fields) < 2:
                    continue

                side_to_move = fields[1]

                if side_to_move not in ("w", "b"):
                    continue

                sign = 1 if side_to_move == "w" else -1

                cp_white = pv.get("cp")
                mate_white = pv.get("mate")

                target_cp_stm = (
                    sign * cp_white
                    if cp_white is not None
                    else None
                )

                target_mate_stm = (
                    sign * mate_white
                    if mate_white is not None
                    else None
                )

                writer.writerow([
                    fen,
                    side_to_move,
                    cp_white,
                    mate_white,
                    target_cp_stm,
                    target_mate_stm,
                    depth,
                    evaluation.get("knodes"),
                    piece_count_from_fen(fen),
                ])

                kept += 1

                if kept % 10_000 == 0:
                    elapsed = time.time() - start

                    print(
                        f"kept={kept:,} "
                        f"scanned={scanned:,} "
                        f"rate={scanned / elapsed:,.0f} records/s"
                    )

                if kept >= args.limit:
                    break

        finally:
            process.terminate()
            process.wait()

    elapsed = time.time() - start

    print()
    print(f"Finished.")
    print(f"Scanned: {scanned:,}")
    print(f"Kept:    {kept:,}")
    print(f"Time:    {elapsed:.1f}s")


if __name__ == "__main__":
    main()

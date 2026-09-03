import argparse
import csv
import json
import random
import subprocess
import time
from collections import Counter
from pathlib import Path


def get_best_evaluation(record):
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
    board = fen.split()[0]

    return sum(
        character.lower() in "pnbrqk"
        for character in board
    )


def cp_bucket(cp):
    absolute = abs(cp)

    if absolute < 50:
        return "0-49"

    if absolute < 150:
        return "50-149"

    if absolute < 400:
        return "150-399"

    if absolute < 1000:
        return "400-999"

    if absolute < 2000:
        return "1000-1999"

    return "2000+"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--validation-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--stats-output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--validation-size",
        type=int,
        default=250_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    rng = random.Random(args.seed)

    reservoir = []

    scanned = 0
    usable = 0
    cp_positions = 0
    mate_positions = 0

    side_counts = Counter()
    piece_counts = Counter()
    depth_counts = Counter()
    cp_buckets = Counter()

    start = time.time()

    source = args.input.resolve()

    process = subprocess.Popen(
        ["zstd", "-dc", str(source)],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1024 * 1024,
    )

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

            fields = fen.split()

            if len(fields) < 2:
                continue

            side_to_move = fields[1]

            if side_to_move not in ("w", "b"):
                continue

            result = get_best_evaluation(record)

            if result is None:
                continue

            evaluation, pv = result

            cp_white = pv.get("cp")
            mate_white = pv.get("mate")

            if cp_white is None and mate_white is None:
                continue

            usable += 1

            depth = evaluation.get("depth", -1)
            knodes = evaluation.get("knodes")
            pieces = piece_count_from_fen(fen)

            side_counts[side_to_move] += 1
            piece_counts[pieces] += 1
            depth_counts[depth] += 1

            if cp_white is not None:
                cp_positions += 1
                cp_buckets[cp_bucket(cp_white)] += 1
            else:
                mate_positions += 1

            row = (
                fen,
                side_to_move,
                cp_white,
                mate_white,
                depth,
                knodes,
                pieces,
            )

            # Uniform reservoir sampling.
            #
            # After seeing n usable positions, every position
            # has equal probability K/n of being in the reservoir.
            if usable <= args.validation_size:
                reservoir.append(row)

            else:
                replacement = rng.randrange(usable)

                if replacement < args.validation_size:
                    reservoir[replacement] = row

            if scanned % 1_000_000 == 0:
                elapsed = time.time() - start

                print(
                    f"scanned={scanned:,} "
                    f"usable={usable:,} "
                    f"rate={scanned / elapsed:,.0f} records/s "
                    f"elapsed={elapsed / 60:.1f}m"
                )

    finally:
        process.stdout.close()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"zstd exited with code {return_code}"
        )

    elapsed = time.time() - start

    args.validation_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.validation_output.open(
        "w",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "fen",
            "side_to_move",
            "cp_white",
            "mate_white",
            "depth",
            "knodes",
            "piece_count",
        ])

        writer.writerows(reservoir)

    stats = {
        "scanned": scanned,
        "usable": usable,
        "cp_positions": cp_positions,
        "mate_positions": mate_positions,
        "mate_fraction": (
            mate_positions / usable
            if usable
            else 0
        ),
        "elapsed_seconds": elapsed,
        "records_per_second": (
            scanned / elapsed
            if elapsed
            else 0
        ),
        "side_to_move": dict(
            sorted(side_counts.items())
        ),
        "piece_count": {
            str(key): value
            for key, value
            in sorted(piece_counts.items())
        },
        "depth": {
            str(key): value
            for key, value
            in sorted(depth_counts.items())
        },
        "absolute_cp_buckets": {
            key: cp_buckets[key]
            for key in [
                "0-49",
                "50-149",
                "150-399",
                "400-999",
                "1000-1999",
                "2000+",
            ]
        },
        "validation_size": len(reservoir),
        "seed": args.seed,
    }

    args.stats_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.stats_output.open("w") as file:
        json.dump(
            stats,
            file,
            indent=2,
        )

    print()
    print("Finished full scan")
    print("------------------")
    print(f"Scanned:        {scanned:,}")
    print(f"Usable:         {usable:,}")
    print(f"CP positions:   {cp_positions:,}")
    print(f"Mate positions: {mate_positions:,}")
    print(
        f"Mate fraction:  "
        f"{100 * stats['mate_fraction']:.2f}%"
    )
    print(
        f"Time:           "
        f"{elapsed / 60:.1f} minutes"
    )
    print(
        f"Rate:           "
        f"{stats['records_per_second']:,.0f} records/s"
    )
    print(
        f"Validation:     "
        f"{len(reservoir):,} positions"
    )

    print()
    print(
        f"Validation file: {args.validation_output}"
    )
    print(
        f"Statistics file: {args.stats_output}"
    )


if __name__ == "__main__":
    main()
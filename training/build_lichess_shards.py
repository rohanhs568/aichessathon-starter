import argparse
import csv
import json
import subprocess
import time
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


def load_validation_fens(path):
    fens = set()

    with path.open(newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            fens.add(row["fen"])

    return fens


def open_shard(output_dir, index, threads):
    final_path = output_dir / f"lichess_train_{index:04d}.tsv.zst"
    temporary_path = output_dir / f"lichess_train_{index:04d}.tsv.zst.tmp"

    process = subprocess.Popen(
        [
            "zstd",
            f"-T{threads}",
            "-3",
            "-q",
            "-f",
            "-o",
            str(temporary_path),
        ],
        stdin=subprocess.PIPE,
        text=True,
    )

    return process, temporary_path, final_path


def close_shard(process, temporary_path, final_path):
    process.stdin.close()

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"zstd compression failed with code {return_code}"
        )

    temporary_path.rename(final_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--validation",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=1_000_000,
    )

    parser.add_argument(
        "--compression-threads",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading permanent validation FENs...")

    validation_fens = load_validation_fens(
        args.validation
    )

    print(
        f"Validation FENs: {len(validation_fens):,}"
    )

    source = args.input.resolve()

    input_process = subprocess.Popen(
        ["zstd", "-dc", str(source)],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1024 * 1024,
    )

    scanned = 0
    written = 0
    validation_skipped = 0

    cp_count = 0
    mate_count = 0

    shard_index = 0
    records_in_shard = 0

    output_process = None
    temporary_path = None
    final_path = None

    start = time.time()

    try:
        for line in input_process.stdout:
            scanned += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            fen = record.get("fen")

            if not fen:
                continue

            # Exclude every occurrence of a permanent
            # validation position from training.
            if fen in validation_fens:
                validation_skipped += 1
                continue

            result = get_best_evaluation(record)

            if result is None:
                continue

            evaluation, pv = result

            cp = pv.get("cp")
            mate = pv.get("mate")

            if cp is None and mate is None:
                continue

            if output_process is None:
                (
                    output_process,
                    temporary_path,
                    final_path,
                ) = open_shard(
                    args.output_dir,
                    shard_index,
                    args.compression_threads,
                )

            cp_text = "" if cp is None else str(cp)
            mate_text = "" if mate is None else str(mate)

            depth = evaluation.get("depth", -1)
            knodes = evaluation.get("knodes", -1)
            piece_count = piece_count_from_fen(fen)

            output_process.stdin.write(
                f"{fen}\t"
                f"{cp_text}\t"
                f"{mate_text}\t"
                f"{depth}\t"
                f"{knodes}\t"
                f"{piece_count}\n"
            )

            written += 1
            records_in_shard += 1

            if cp is not None:
                cp_count += 1
            else:
                mate_count += 1

            if records_in_shard >= args.shard_size:
                close_shard(
                    output_process,
                    temporary_path,
                    final_path,
                )

                print(
                    f"completed shard {shard_index:04d}: "
                    f"{records_in_shard:,} positions"
                )

                shard_index += 1
                records_in_shard = 0

                output_process = None
                temporary_path = None
                final_path = None

            if scanned % 1_000_000 == 0:
                elapsed = time.time() - start

                print(
                    f"scanned={scanned:,} "
                    f"written={written:,} "
                    f"rate={scanned / elapsed:,.0f}/s "
                    f"elapsed={elapsed / 60:.1f}m"
                )

    finally:
        if output_process is not None:
            close_shard(
                output_process,
                temporary_path,
                final_path,
            )

        input_process.stdout.close()
        input_code = input_process.wait()

    if input_code != 0:
        raise RuntimeError(
            f"input zstd exited with code {input_code}"
        )

    elapsed = time.time() - start

    metadata = {
        "scanned": scanned,
        "written": written,
        "validation_skipped": validation_skipped,
        "cp_positions": cp_count,
        "mate_positions": mate_count,
        "shards": shard_index + (
            1 if records_in_shard else 0
        ),
        "shard_size": args.shard_size,
        "elapsed_seconds": elapsed,
        "records_per_second": (
            scanned / elapsed
            if elapsed
            else 0
        ),
        "columns": [
            "fen",
            "cp_white",
            "mate_white",
            "depth",
            "knodes",
            "piece_count",
        ],
    }

    metadata_path = (
        args.output_dir / "metadata.json"
    )

    with metadata_path.open("w") as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print()
    print("Finished")
    print("--------")
    print(f"Scanned:            {scanned:,}")
    print(f"Training positions: {written:,}")
    print(
        f"Validation skipped: {validation_skipped:,}"
    )
    print(f"CP positions:       {cp_count:,}")
    print(f"Mate positions:     {mate_count:,}")
    print(
        f"Runtime:            {elapsed / 60:.1f} min"
    )
    print(
        f"Rate:               "
        f"{metadata['records_per_second']:,.0f}/s"
    )
    print(f"Output:              {args.output_dir}")


if __name__ == "__main__":
    main()
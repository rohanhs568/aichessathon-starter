"""Annotate sampled chess positions with Stockfish evaluations."""

import argparse
import csv
import time

import chess
import chess.engine


STOCKFISH_PATH = "/usr/games/stockfish"
MATE_SCORE = 10_000


def evaluate_position(engine, fen, depth):
    board = chess.Board(fen)

    info = engine.analyse(
        board,
        chess.engine.Limit(depth=depth),
    )

    score = info["score"].pov(board.turn)

    mate = score.mate()

    cp = score.score(
        mate_score=MATE_SCORE
    )

    return cp, mate


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="training/data/positions.csv",
    )

    parser.add_argument(
        "--output",
        default="training/data/labelled_positions.csv",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=14,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    with open(
        args.input,
        "r",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    if args.limit is not None:
        rows = rows[:args.limit]

    fieldnames = list(rows[0].keys()) + [
        "stockfish_cp",
        "mate",
    ]

    start_time = time.monotonic()

    with chess.engine.SimpleEngine.popen_uci(
        STOCKFISH_PATH
    ) as engine:

        with open(
            args.output,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for i, row in enumerate(rows, start=1):
                cp, mate = evaluate_position(
                    engine,
                    row["fen"],
                    args.depth,
                )

                output_row = dict(row)
                output_row["stockfish_cp"] = cp
                output_row["mate"] = (
                    "" if mate is None else mate
                )

                writer.writerow(output_row)

                if i % 100 == 0 or i == len(rows):
                    elapsed = time.monotonic() - start_time
                    rate = i / elapsed

                    print(
                        f"{i}/{len(rows)} "
                        f"rate={rate:.2f} positions/s "
                        f"cp={cp} "
                        f"mate={mate}"
                    )


if __name__ == "__main__":
    main()
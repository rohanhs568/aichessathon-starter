"""Sample diverse chess positions from PGN games."""

import argparse
import csv
import glob
import random

import chess
import chess.pgn


def phase_from_ply(ply):
    """
    Crude phase labels for dataset inspection.

    These are NOT used by the chess engine.
    """
    if ply <= 20:
        return "opening"
    elif ply <= 60:
        return "middlegame"
    else:
        return "late"


def position_key(board):
    """
    Ignore halfmove/fullmove counters when deduplicating.

    Keep:
        piece placement
        side to move
        castling rights
        en passant square
    """
    return " ".join(board.fen().split()[:4])


def extract_positions(game):
    """
    Return candidate non-terminal positions from one game.
    """

    board = game.board()
    positions = []

    for ply, move in enumerate(game.mainline_moves(), start=1):
        board.push(move)

        # Don't train our static evaluator on terminal positions.
        if board.is_game_over(claim_draw=True):
            continue

        # Skip the extremely early opening.
        if ply < 6:
            continue

        positions.append(
            {
                "fen": board.fen(),
                "ply": ply,
                "phase": phase_from_ply(ply),
                "piece_count": len(board.piece_map()),
                "result": game.headers.get("Result", "*"),
            }
        )

    return positions


def sample_game_positions(positions, per_game, rng):
    """
    Try to obtain positions from different parts of the game
    rather than taking several adjacent positions.
    """

    if len(positions) <= per_game:
        return positions

    buckets = {
        "opening": [],
        "middlegame": [],
        "late": [],
    }

    for position in positions:
        buckets[position["phase"]].append(position)

    chosen = []

    # First try to get representation from every phase.
    for phase in ("opening", "middlegame", "late"):
        if buckets[phase]:
            chosen.append(rng.choice(buckets[phase]))

    # Fill remaining slots randomly from positions not yet chosen.
    remaining = [
        position
        for position in positions
        if position not in chosen
    ]

    rng.shuffle(remaining)

    needed = per_game - len(chosen)

    chosen.extend(remaining[:needed])

    return chosen


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="training/data/raw/*.pgn",
    )

    parser.add_argument(
        "--output",
        default="training/data/positions.csv",
    )

    parser.add_argument(
        "--target",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--per-game",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    rng = random.Random(args.seed)

    pgn_files = sorted(glob.glob(args.input))

    if not pgn_files:
        raise FileNotFoundError(
            f"No PGN files found matching {args.input}"
        )

    seen = set()
    dataset = []

    games_read = 0

    for pgn_path in pgn_files:
        print(f"Reading {pgn_path}")

        with open(
            pgn_path,
            "r",
            encoding="utf-8",
            errors="replace",
        ) as pgn:

            while len(dataset) < args.target:
                game = chess.pgn.read_game(pgn)

                if game is None:
                    break

                games_read += 1

                positions = extract_positions(game)

                sampled = sample_game_positions(
                    positions,
                    args.per_game,
                    rng,
                )

                for position in sampled:
                    board = chess.Board(position["fen"])
                    key = position_key(board)

                    if key in seen:
                        continue

                    seen.add(key)
                    dataset.append(position)

                    if len(dataset) >= args.target:
                        break

                if games_read % 1000 == 0:
                    print(
                        f"games={games_read} "
                        f"positions={len(dataset)}"
                    )

        if len(dataset) >= args.target:
            break

    rng.shuffle(dataset)

    with open(
        args.output,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "fen",
                "ply",
                "phase",
                "piece_count",
                "result",
            ],
        )

        writer.writeheader()
        writer.writerows(dataset)

    print()
    print(f"Games read: {games_read}")
    print(f"Unique positions: {len(dataset)}")

    phases = {
        "opening": 0,
        "middlegame": 0,
        "late": 0,
    }

    for row in dataset:
        phases[row["phase"]] += 1

    print(f"Opening: {phases['opening']}")
    print(f"Middlegame: {phases['middlegame']}")
    print(f"Late: {phases['late']}")


if __name__ == "__main__":
    main()
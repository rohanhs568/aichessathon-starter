"""Paired arena over a file of starting FENs.

Each FEN is played twice with colours reversed. This is much more informative
than repeating the normal starting position, and is useful for A/B testing two
deterministic engines.

FEN file format:
    one FEN per line
    blank lines and lines beginning with # are ignored
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import chess
import chess.pgn

from harness.referee import FAILED_TERMINATIONS, play_match
from harness.rules import PLY_CAP
from harness.sandbox import local


def load_fens(path: Path) -> list[str]:
    fens = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        board = chess.Board(text)
        if board.is_game_over(claim_draw=True):
            continue
        fens.append(board.fen())
    if not fens:
        raise SystemExit(f"No playable FENs found in {path}")
    return fens


def labelled_pgn(
    raw_pgn: str,
    game_number: int,
    white_name: str,
    black_name: str,
    start_fen: str,
) -> str:
    game = chess.pgn.read_game(io.StringIO(raw_pgn))
    if game is None:
        return raw_pgn

    game.headers["Event"] = "AI Chessathon paired FEN arena"
    game.headers["Round"] = str(game_number)
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["SetUp"] = "1"
    game.headers["FEN"] = start_fen
    return str(game)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--fens", type=Path, required=True)
    parser.add_argument("--base-ms", type=int, default=10_000)
    parser.add_argument("--increment-ms", type=int, default=100)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument("--pgn-out", type=Path, default=Path("paired_arena.pgn"))
    args = parser.parse_args()

    agent = args.agent.resolve()
    opponent = args.opponent.resolve()
    fens = load_fens(args.fens)

    wins = draws = losses = 0
    terminations: dict[str, int] = {}
    games: list[str] = []

    game_number = 0

    for fen_index, fen in enumerate(fens, start=1):
        for agent_white in (True, False):
            game_number += 1

            if agent_white:
                white_path, black_path = agent, opponent
                white_name, black_name = args.agent.as_posix(), args.opponent.as_posix()
            else:
                white_path, black_path = opponent, agent
                white_name, black_name = args.opponent.as_posix(), args.agent.as_posix()

            outcome = play_match(
                local(white_path),
                local(black_path),
                args.base_ms,
                args.increment_ms,
                ply_cap=args.ply_cap,
                start_fen=fen,
            )

            games.append(
                labelled_pgn(
                    outcome.pgn,
                    game_number,
                    white_name,
                    black_name,
                    fen,
                )
            )

            terminations[outcome.termination] = (
                terminations.get(outcome.termination, 0) + 1
            )

            if outcome.result in {"draw", "void"}:
                draws += 1
            elif (outcome.result == "white") == agent_white:
                wins += 1
            else:
                losses += 1

            print(
                f"fen {fen_index}/{len(fens)} "
                f"game {1 if agent_white else 2}/2: "
                f"{outcome.result} by {outcome.termination}"
            )

    args.pgn_out.write_text("\n\n".join(games) + "\n", encoding="utf-8")

    total = wins + draws + losses
    score = (wins + 0.5 * draws) / total

    print()
    print(f"{args.agent} vs {args.opponent} over {total} paired games")
    print(f"+{wins} ={draws} -{losses}, score {score:.1%}")
    print(
        "terminations: "
        + ", ".join(f"{name} {count}" for name, count in terminations.items())
    )
    print(f"PGNs saved to: {args.pgn_out}")

    broken = {
        name: count
        for name, count in terminations.items()
        if name in FAILED_TERMINATIONS
    }
    if broken:
        raise SystemExit(
            "agent failure: "
            + ", ".join(f"{name} {count}" for name, count in broken.items())
        )


if __name__ == "__main__":
    main()

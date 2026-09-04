"""Local Stockfish opponent for Chessathon testing.

LOCAL BENCHMARKING ONLY. Do not package this baseline with the submission.

Environment variables:
    STOCKFISH_PATH       Path to Stockfish binary.
    STOCKFISH_ELO        Limited-strength Elo. Default 2000.
                         Set to 0 for unrestricted Stockfish.
    STOCKFISH_MOVE_MS    Think time per move. Default 50 ms.
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path

import chess
import chess.engine


def _find_stockfish() -> str:
    candidates = [
        os.environ.get("STOCKFISH_PATH"),
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
    ]

    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate

    raise FileNotFoundError(
        "Stockfish binary not found. Install it with "
        "`sudo apt install stockfish`, or set STOCKFISH_PATH."
    )


ENGINE = chess.engine.SimpleEngine.popen_uci(_find_stockfish())

options: dict[str, object] = {}

if "Threads" in ENGINE.options:
    options["Threads"] = 1

if "Hash" in ENGINE.options:
    options["Hash"] = 64

elo = int(os.environ.get("STOCKFISH_ELO", "2000"))

if elo > 0 and "UCI_LimitStrength" in ENGINE.options and "UCI_Elo" in ENGINE.options:
    options["UCI_LimitStrength"] = True

    elo_option = ENGINE.options["UCI_Elo"]
    min_elo = int(elo_option.min or elo)
    max_elo = int(elo_option.max or elo)
    options["UCI_Elo"] = max(min_elo, min(max_elo, elo))

elif elo > 0 and "Skill Level" in ENGINE.options:
    # Fallback for older Stockfish builds.
    options["Skill Level"] = 10

elif elo <= 0 and "UCI_LimitStrength" in ENGINE.options:
    options["UCI_LimitStrength"] = False

if options:
    ENGINE.configure(options)


def _close_engine() -> None:
    try:
        ENGINE.quit()
    except Exception:
        pass


atexit.register(_close_engine)


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)

    if board.is_game_over():
        raise ValueError("get_move called on terminal position")

    requested_ms = int(os.environ.get("STOCKFISH_MOVE_MS", "50"))

    think_ms = max(
        5,
        min(
            requested_ms,
            max(5, int(time_left_ms * 0.05)),
        ),
    )

    result = ENGINE.play(
        board,
        chess.engine.Limit(time=think_ms / 1000.0),
    )

    if result.move is None:
        raise RuntimeError("Stockfish returned no move")

    return result.move.uci()

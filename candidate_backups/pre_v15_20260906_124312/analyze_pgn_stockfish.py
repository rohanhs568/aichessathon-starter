"""Analyse Chessathon PGNs with full-strength Stockfish.

For every move, this script measures the centipawn loss relative to
Stockfish's best move from the mover's perspective. It writes:
    1. a CSV containing every analysed move;
    2. a text summary with ACPL and mistake/blunder counts by side.

Example:
    python training/analyze_pgn_stockfish.py \
        nn_vs_stockfish_full.pgn \
        --stockfish /usr/games/stockfish \
        --depth 16 \
        --csv-out analysis.csv \
        --summary-out analysis.txt
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import chess
import chess.engine
import chess.pgn

MATE_SCORE = 100_000
CP_CLAMP = 1_000


def clamp_cp(value: int) -> int:
    return max(-CP_CLAMP, min(CP_CLAMP, value))


def numeric_score(score: chess.engine.PovScore, colour: chess.Color) -> int:
    value = score.pov(colour).score(mate_score=MATE_SCORE)
    if value is None:
        raise RuntimeError("Stockfish returned a score that could not be converted")
    return int(value)


def phase(board: chess.Board) -> str:
    pieces = len(board.piece_map())
    if pieces >= 25:
        return "opening/middlegame"
    if pieces >= 13:
        return "middlegame"
    return "endgame"


def classify_loss(cp_loss: int) -> str:
    if cp_loss >= 300:
        return "blunder"
    if cp_loss >= 150:
        return "major_mistake"
    if cp_loss >= 75:
        return "mistake"
    if cp_loss >= 30:
        return "inaccuracy"
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pgn", type=Path)
    parser.add_argument("--stockfish", type=Path, default=Path("/usr/games/stockfish"))
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=256)
    parser.add_argument("--csv-out", type=Path, default=Path("stockfish_analysis.csv"))
    parser.add_argument("--summary-out", type=Path, default=Path("stockfish_analysis.txt"))
    args = parser.parse_args()

    if not args.pgn.is_file():
        raise SystemExit(f"PGN not found: {args.pgn}")
    if not args.stockfish.is_file():
        raise SystemExit(f"Stockfish not found: {args.stockfish}")

    engine = chess.engine.SimpleEngine.popen_uci(str(args.stockfish))
    options: dict[str, object] = {}
    if "Threads" in engine.options:
        options["Threads"] = args.threads
    if "Hash" in engine.options:
        options["Hash"] = args.hash_mb
    if "UCI_LimitStrength" in engine.options:
        options["UCI_LimitStrength"] = False
    if options:
        engine.configure(options)

    rows: list[dict[str, object]] = []
    totals = defaultdict(lambda: {
        "moves": 0,
        "cp_loss": 0,
        "inaccuracy": 0,
        "mistake": 0,
        "major_mistake": 0,
        "blunder": 0,
    })

    game_index = 0

    try:
        with args.pgn.open("r", encoding="utf-8") as handle:
            while True:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break

                game_index += 1
                board = game.board()

                white_name = game.headers.get("White", "White")
                black_name = game.headers.get("Black", "Black")

                for ply, move in enumerate(game.mainline_moves(), start=1):
                    mover = board.turn
                    mover_name = white_name if mover == chess.WHITE else black_name
                    move_number = board.fullmove_number
                    san = board.san(move)
                    current_phase = phase(board)

                    best_info = engine.analyse(
                        board,
                        chess.engine.Limit(depth=args.depth),
                    )
                    best_score_raw = numeric_score(best_info["score"], mover)
                    best_score = clamp_cp(best_score_raw)
                    best_move = best_info.get("pv", [None])[0]

                    board.push(move)

                    played_info = engine.analyse(
                        board,
                        chess.engine.Limit(depth=args.depth),
                    )
                    played_score_raw = numeric_score(played_info["score"], mover)
                    played_score = clamp_cp(played_score_raw)

                    # Mate scores are synthetic sentinels, not literal CP.
                    # Clamp each eval before differencing so one mate flip
                    # cannot dominate ACPL.
                    cp_loss = max(0, best_score - played_score)
                    label = classify_loss(cp_loss)

                    bucket = totals[mover_name]
                    bucket["moves"] += 1
                    bucket["cp_loss"] += cp_loss
                    if label != "ok":
                        bucket[label] += 1

                    rows.append({
                        "game": game_index,
                        "ply": ply,
                        "move_number": move_number,
                        "side": "white" if mover == chess.WHITE else "black",
                        "player": mover_name,
                        "phase": current_phase,
                        "move": san,
                        "best_move": (
                            best_move.uci() if best_move is not None else ""
                        ),
                        "best_eval_raw_cp": best_score_raw,
                        "played_eval_raw_cp": played_score_raw,
                        "best_eval_cp": best_score,
                        "played_eval_cp": played_score,
                        "cp_loss": cp_loss,
                        "classification": label,
                    })

                    if cp_loss >= 150:
                        print(
                            f"game {game_index:>2} ply {ply:>3} "
                            f"{mover_name}: {san:<8} "
                            f"loss={cp_loss:>4}cp "
                            f"best={best_move}"
                        )

    finally:
        engine.quit()

    with args.csv_out.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "game",
            "ply",
            "move_number",
            "side",
            "player",
            "phase",
            "move",
            "best_move",
            "best_eval_raw_cp",
            "played_eval_raw_cp",
            "best_eval_cp",
            "played_eval_cp",
            "cp_loss",
            "classification",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"PGN: {args.pgn}",
        f"Stockfish: {args.stockfish}",
        f"Analysis depth: {args.depth}",
        f"Games: {game_index}",
        "",
    ]

    for player, data in totals.items():
        moves = int(data["moves"])
        acpl = data["cp_loss"] / moves if moves else 0.0
        lines.extend([
            player,
            f"  analysed moves: {moves}",
            f"  ACPL (evals clamped to +/-{CP_CLAMP}cp): {acpl:.1f}",
            f"  inaccuracies >=30cp: {data['inaccuracy']}",
            f"  mistakes >=75cp: {data['mistake']}",
            f"  major mistakes >=150cp: {data['major_mistake']}",
            f"  blunders >=300cp: {data['blunder']}",
            "",
        ])

    args.summary_out.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("\n".join(lines))
    print(f"CSV saved to: {args.csv_out}")
    print(f"Summary saved to: {args.summary_out}")


if __name__ == "__main__":
    main()

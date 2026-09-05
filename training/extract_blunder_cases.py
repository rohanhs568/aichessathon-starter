"""Extract search-regression positions from PGNs using Stockfish.

For every played move, evaluate:
  * Stockfish best move from the original root position
  * the actually played move with root_moves=[played_move]

Scores are always converted to the player-who-moved's point of view. WDL
expectation loss is the primary selector; CP loss is secondary.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import chess
import chess.engine
import chess.pgn

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


def configure(engine, hash_mb):
    opts = {}
    if "Threads" in engine.options:
        opts["Threads"] = 1
    if "Hash" in engine.options:
        opts["Hash"] = hash_mb
    if "UCI_ShowWDL" in engine.options:
        opts["UCI_ShowWDL"] = True
    if opts:
        engine.configure(opts)


def score_record(info, color, ply):
    score = info["score"].pov(color)
    cp = score.score(mate_score=100_000)
    if "wdl" in info:
        wdl = info["wdl"].pov(color)
    else:
        wdl = score.wdl(model="sf", ply=ply)
    return {
        "cp": int(cp),
        "mate": score.mate(),
        "score": str(score),
        "expectation": float(wdl.expectation()),
        "pv": " ".join(m.uci() for m in info.get("pv", [])[:12]),
    }


def material_balance(board, color):
    return sum(
        value * (len(board.pieces(piece, color)) - len(board.pieces(piece, not color)))
        for piece, value in PIECE_VALUE.items()
    )


def classify(board, move, best, played, cp_loss, exp_loss):
    tags = []
    piece_count = len(board.piece_map())
    if piece_count <= 10:
        tags.append("endgame")
    elif piece_count <= 16:
        tags.append("late")
    else:
        tags.append("middlegame")

    if board.is_capture(move):
        tags.append("capture")
    if board.gives_check(move):
        tags.append("check")
    if move.promotion:
        tags.append("promotion")

    if cp_loss >= 300:
        tags.append("blunder")
    elif cp_loss >= 150:
        tags.append("major")
    elif cp_loss >= 75:
        tags.append("mistake")

    if best["expectation"] >= 0.75 and played["expectation"] < 0.60:
        tags.append("conversion_failure")
    if best["cp"] >= 100 and played["cp"] <= -100:
        tags.append("sign_flip")
    if played["mate"] is not None and played["mate"] < 0 and not (best["mate"] is not None and best["mate"] < 0):
        tags.append("mate_blunder")

    # Simple immediate material context. This is not SEE and is not used to
    # decide whether a move is a true sacrifice; it just helps filter cases.
    before = material_balance(board, board.turn)
    board.push(move)
    after = -material_balance(board, board.turn)  # root player's perspective after turn flips
    board.pop()
    if after < before - 150:
        tags.append("immediate_material_drop")

    return "+".join(tags)


def iter_games(paths):
    for path in paths:
        with path.open(errors="replace") as f:
            game_index = 0
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                game_index += 1
                yield path, game_index, game


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pgn", nargs="+", type=Path)
    parser.add_argument("--stockfish", default="/usr/games/stockfish")
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--hash", type=int, default=128)
    parser.add_argument("--min-cp-loss", type=float, default=50.0)
    parser.add_argument("--min-expectation-loss", type=float, default=0.02)
    parser.add_argument("--max-cases", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("training/data/search_regression_cases.csv"))
    args = parser.parse_args()

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    configure(engine, args.hash)
    rows = []

    try:
        for path, game_index, game in iter_games(args.pgn):
            board = game.board()
            ply = 0
            for move in game.mainline_moves():
                ply += 1
                mover = board.turn
                fen = board.fen()
                san = board.san(move)

                best_info = engine.analyse(board, chess.engine.Limit(depth=args.depth))
                best = score_record(best_info, mover, board.ply())
                best_move = best_info.get("pv", [None])[0]

                if best_move == move:
                    played = best
                else:
                    played_info = engine.analyse(
                        board,
                        chess.engine.Limit(depth=args.depth),
                        root_moves=[move],
                    )
                    played = score_record(played_info, mover, board.ply())

                cp_loss = max(0.0, float(best["cp"] - played["cp"]))
                exp_loss = max(0.0, float(best["expectation"] - played["expectation"]))

                if cp_loss >= args.min_cp_loss or exp_loss >= args.min_expectation_loss:
                    category = classify(board, move, best, played, cp_loss, exp_loss)
                    rows.append({
                        "id": f"{path.stem}_g{game_index}_ply{ply}",
                        "fen": fen,
                        "category": category,
                        "source": str(path),
                        "game": game_index,
                        "ply": ply,
                        "side": "white" if mover == chess.WHITE else "black",
                        "played_move": move.uci(),
                        "played_san": san,
                        "sf_best_move": best_move.uci() if best_move else "",
                        "sf_best_score": best["score"],
                        "sf_played_score": played["score"],
                        "cp_loss": cp_loss,
                        "expectation_loss": exp_loss,
                        "sf_best_pv": best["pv"],
                        "sf_played_pv": played["pv"],
                    })
                    print(
                        f"{rows[-1]['id']}: {san} loss={cp_loss:.0f}cp "
                        f"exp={exp_loss:.3f} {category}"
                    )

                board.push(move)

        rows.sort(key=lambda r: (r["expectation_loss"], min(r["cp_loss"], 2000)), reverse=True)
        rows = rows[: args.max_cases]
    finally:
        engine.quit()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote {len(rows)} regression cases to {args.out}")


if __name__ == "__main__":
    main()

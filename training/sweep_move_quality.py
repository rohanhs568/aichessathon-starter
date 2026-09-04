"""Fast A/B sweep for Chessathon engine configurations.

This is intentionally much faster than playing full matches. For each test FEN:
  1. ask each engine configuration for one move using the real harness protocol;
  2. let full-strength Stockfish score the best move and the played move;
  3. report centipawn loss, mistakes, blunders, and a ranked summary.

Default configurations:
    submitted_v1     /tmp/learned_v1
    search_only      current agent.py with MATERIAL_BLEND=0.00
    blend_15         current agent.py with MATERIAL_BLEND=0.15
    blend_25         current agent.py with MATERIAL_BLEND=0.25
    blend_35         current agent.py with MATERIAL_BLEND=0.35

Run from the repository root:
    python training/sweep_move_quality.py

Outputs:
    sweep_move_quality.csv
    sweep_move_quality.txt
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import defaultdict
from pathlib import Path

import chess
import chess.engine

from harness.sandbox import AgentFailure, local


MATE_CP = 2000


def load_fens(path: Path, max_fens: int) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"FEN file not found: {path}")

    fens: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue

        try:
            board = chess.Board(text)
        except ValueError:
            continue

        if board.is_game_over(claim_draw=True):
            continue

        fens.append(board.fen())
        if len(fens) >= max_fens:
            break

    if not fens:
        raise SystemExit(f"No playable FENs found in {path}")

    return fens


def score_cp(info: dict, colour: chess.Color) -> int:
    value = info["score"].pov(colour).score(mate_score=MATE_CP)
    if value is None:
        raise RuntimeError("Could not convert Stockfish score")
    return int(value)


def classify(cp_loss: int) -> str:
    if cp_loss >= 300:
        return "blunder"
    if cp_loss >= 150:
        return "major_mistake"
    if cp_loss >= 75:
        return "mistake"
    if cp_loss >= 30:
        return "inaccuracy"
    return "ok"


def run_agent_moves(
    directory: Path,
    fens: list[str],
    time_left_ms: int,
    material_blend: float | None,
) -> list[str | None]:
    old_blend = os.environ.get("MATERIAL_BLEND")

    if material_blend is None:
        os.environ.pop("MATERIAL_BLEND", None)
    else:
        os.environ["MATERIAL_BLEND"] = str(material_blend)

    agent = local(directory)
    moves: list[str | None] = []

    try:
        agent.start(init_budget_s=60.0)

        for index, fen in enumerate(fens, start=1):
            try:
                move_text = agent.move(fen, time_left_ms)
                board = chess.Board(fen)
                move = chess.Move.from_uci(move_text)

                if move not in board.legal_moves:
                    print(
                        f"  FEN {index}: illegal move {move_text}",
                        flush=True,
                    )
                    moves.append(None)
                else:
                    moves.append(move_text)

            except (AgentFailure, ValueError) as exc:
                print(
                    f"  FEN {index}: engine failure {exc}",
                    flush=True,
                )
                moves.append(None)

    finally:
        agent.stop()

        if old_blend is None:
            os.environ.pop("MATERIAL_BLEND", None)
        else:
            os.environ["MATERIAL_BLEND"] = old_blend

    return moves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fens",
        type=Path,
        default=Path("training/data/test_fens.txt"),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("/tmp/learned_v1"),
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--stockfish",
        type=Path,
        default=Path("/usr/games/stockfish"),
    )
    parser.add_argument(
        "--max-fens",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--time-left-ms",
        type=int,
        default=3000,
        help=(
            "Clock passed to each candidate move. With the current time manager, "
            "3000ms means roughly 225ms search per position."
        ),
    )
    parser.add_argument(
        "--stockfish-depth",
        type=int,
        default=13,
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("sweep_move_quality.csv"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("sweep_move_quality.txt"),
    )
    args = parser.parse_args()

    if not args.stockfish.is_file():
        raise SystemExit(f"Stockfish not found: {args.stockfish}")

    if not args.baseline.is_dir():
        raise SystemExit(
            f"Submitted V1 baseline not found: {args.baseline}\n"
            "Expected the saved baseline at /tmp/learned_v1."
        )

    fens = load_fens(args.fens, args.max_fens)

    configs = [
        ("submitted_v1", args.baseline, None),
        ("search_only", args.current, 0.00),
        ("blend_15", args.current, 0.15),
        ("blend_25", args.current, 0.25),
        ("blend_35", args.current, 0.35),
    ]

    print(
        f"Testing {len(configs)} configurations on {len(fens)} positions",
        flush=True,
    )
    print(
        f"Candidate clock: {args.time_left_ms}ms | "
        f"Stockfish depth: {args.stockfish_depth}",
        flush=True,
    )
    print()

    candidate_moves: dict[str, list[str | None]] = {}

    for name, directory, blend in configs:
        detail = (
            "original submitted V1"
            if blend is None
            else f"MATERIAL_BLEND={blend:.2f}"
        )
        print(f"[{name}] {detail}", flush=True)
        candidate_moves[name] = run_agent_moves(
            directory=directory,
            fens=fens,
            time_left_ms=args.time_left_ms,
            material_blend=blend,
        )
        legal_count = sum(move is not None for move in candidate_moves[name])
        print(
            f"  legal moves: {legal_count}/{len(fens)}",
            flush=True,
        )
        print()

    engine = chess.engine.SimpleEngine.popen_uci(str(args.stockfish))

    options: dict[str, object] = {}
    if "Threads" in engine.options:
        options["Threads"] = 1
    if "Hash" in engine.options:
        options["Hash"] = 256
    if "UCI_LimitStrength" in engine.options:
        options["UCI_LimitStrength"] = False
    if options:
        engine.configure(options)

    rows: list[dict[str, object]] = []
    losses: dict[str, list[int]] = defaultdict(list)
    labels: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    best_move_hits: dict[str, int] = defaultdict(int)
    failures: dict[str, int] = defaultdict(int)

    # Cache played-position analyses because different configs often choose
    # the same move.
    played_cache: dict[tuple[str, str], int] = {}

    try:
        for fen_index, fen in enumerate(fens, start=1):
            board = chess.Board(fen)
            mover = board.turn

            best_info = engine.analyse(
                board,
                chess.engine.Limit(depth=args.stockfish_depth),
            )
            best_score = score_cp(best_info, mover)
            best_pv = best_info.get("pv", [])
            best_move = best_pv[0] if best_pv else None
            best_uci = "" if best_move is None else best_move.uci()

            print(
                f"Stockfish {fen_index}/{len(fens)} "
                f"best={best_uci} eval={best_score:+d}",
                flush=True,
            )

            for name, _directory, _blend in configs:
                move_text = candidate_moves[name][fen_index - 1]

                if move_text is None:
                    failures[name] += 1
                    rows.append(
                        {
                            "fen_index": fen_index,
                            "config": name,
                            "fen": fen,
                            "move": "",
                            "stockfish_best": best_uci,
                            "best_eval_cp": best_score,
                            "played_eval_cp": "",
                            "cp_loss": "",
                            "classification": "failure",
                        }
                    )
                    continue

                move = chess.Move.from_uci(move_text)

                if best_move is not None and move == best_move:
                    best_move_hits[name] += 1

                cache_key = (fen, move_text)
                played_score = played_cache.get(cache_key)

                if played_score is None:
                    after = board.copy(stack=False)
                    after.push(move)

                    played_info = engine.analyse(
                        after,
                        chess.engine.Limit(depth=args.stockfish_depth),
                    )
                    played_score = score_cp(played_info, mover)
                    played_cache[cache_key] = played_score

                cp_loss = max(0, best_score - played_score)
                # Mate scores are represented as +/-2000 above, so one result
                # cannot dominate the average arbitrarily.
                cp_loss = min(cp_loss, 4000)

                label = classify(cp_loss)
                losses[name].append(cp_loss)
                labels[name][label] += 1

                rows.append(
                    {
                        "fen_index": fen_index,
                        "config": name,
                        "fen": fen,
                        "move": move_text,
                        "stockfish_best": best_uci,
                        "best_eval_cp": best_score,
                        "played_eval_cp": played_score,
                        "cp_loss": cp_loss,
                        "classification": label,
                    }
                )

    finally:
        engine.quit()

    with args.csv_out.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "fen_index",
            "config",
            "fen",
            "move",
            "stockfish_best",
            "best_eval_cp",
            "played_eval_cp",
            "cp_loss",
            "classification",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []

    for name, _directory, _blend in configs:
        values = losses[name]
        analysed = len(values)

        if analysed:
            mean_loss = statistics.fmean(values)
            median_loss = statistics.median(values)
            good_rate = sum(value < 30 for value in values) / analysed
            mistake_rate = sum(value >= 75 for value in values) / analysed
            blunder_rate = sum(value >= 300 for value in values) / analysed
            best_rate = best_move_hits[name] / analysed
        else:
            mean_loss = float("inf")
            median_loss = float("inf")
            good_rate = 0.0
            mistake_rate = 1.0
            blunder_rate = 1.0
            best_rate = 0.0

        summary_rows.append(
            {
                "config": name,
                "analysed": analysed,
                "failures": failures[name],
                "mean": mean_loss,
                "median": median_loss,
                "good": good_rate,
                "mistakes": mistake_rate,
                "blunders": blunder_rate,
                "best": best_rate,
            }
        )

    summary_rows.sort(
        key=lambda row: (
            row["failures"],
            row["mean"],
            row["median"],
        )
    )

    lines = [
        "",
        "=" * 86,
        "MOVE-QUALITY SWEEP",
        "=" * 86,
        (
            f"Positions: {len(fens)} | candidate clock: {args.time_left_ms}ms | "
            f"Stockfish depth: {args.stockfish_depth}"
        ),
        "",
        (
            f"{'rank':<5} {'config':<15} {'mean':>8} {'median':>8} "
            f"{'<30cp':>8} {'>=75cp':>8} {'>=300':>8} {'SF best':>8}"
        ),
        "-" * 86,
    ]

    for rank, row in enumerate(summary_rows, start=1):
        lines.append(
            f"{rank:<5} "
            f"{row['config']:<15} "
            f"{row['mean']:>7.1f} "
            f"{row['median']:>7.1f} "
            f"{100 * row['good']:>7.1f}% "
            f"{100 * row['mistakes']:>7.1f}% "
            f"{100 * row['blunders']:>7.1f}% "
            f"{100 * row['best']:>7.1f}%"
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "  lower mean/median centipawn loss is better;",
            "  <30cp is the fraction of moves close to Stockfish;",
            "  >=75cp and >=300cp are mistake/blunder rates;",
            "  SF best is exact agreement with Stockfish's top move.",
            "",
            (
                "Use this sweep to select a candidate quickly. "
                "Then confirm the winner in paired games."
            ),
            "",
            f"Detailed CSV: {args.csv_out}",
        ]
    )

    text = "\n".join(lines)
    print(text)

    args.summary_out.write_text(text + "\n", encoding="utf-8")
    print(f"Summary saved to: {args.summary_out}")


if __name__ == "__main__":
    main()

"""A/B test one isolated search change: V1 quiescence vs V1 quiescence+quiet checks.

Nothing in the evaluator, material weighting, LMR, null move, futility, time
manager, or root search is changed. Both candidates load the exact submitted V1
from /tmp/learned_v1.

The experimental quiescence adds only:
    * one ply of non-capture checking moves at quiet frontier nodes;
    * explicit qsearch ordering that places promotions/captures before quiet checks.

This is an experiment. It does not modify agent.py.

Run:
    python -m training.test_qsearch_v2 \
      --fens training/data/test_fens_60.txt \
      --max-fens 60 \
      --time-left-ms 5000

Outputs:
    qsearch_v2_ab.csv
    qsearch_v2_ab.txt
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
import sys
from pathlib import Path

import chess
import chess.engine


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_agent(name: str, agent_path: Path):
    spec = importlib.util.spec_from_file_location(name, agent_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {agent_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_qsearch_v2(module) -> None:
    """Replace only module.quiescence. Everything else remains submitted V1."""

    def qmove_score(board: chess.Board, move: chess.Move) -> int:
        score = 0

        if move.promotion is not None:
            score += 10_000_000 + module.PIECE_VALUE.get(move.promotion, 0)

        if board.is_capture(move):
            victim, attacker = module.capture_values(board, move)
            score += 1_000_000 + 10 * victim - attacker

        if board.gives_check(move):
            score += 100_000

        return score

    def quiescence(board, alpha, beta, ply, check_budget=1):
        module.NODES += 1
        module.check_time()

        moves = list(board.legal_moves)

        if not moves:
            if board.is_check():
                return -module.MATE + ply
            return 0

        if board.halfmove_clock >= 100:
            return 0

        # If we are in check, stand-pat is illegal. Search every legal evasion.
        if board.is_check():
            moves.sort(
                key=lambda move: qmove_score(board, move),
                reverse=True,
            )

            best_score = -module.INF

            for move in moves:
                board.push(move)
                try:
                    score = -quiescence(
                        board,
                        -beta,
                        -alpha,
                        ply + 1,
                        check_budget,
                    )
                finally:
                    board.pop()

                if score > best_score:
                    best_score = score

                if score > alpha:
                    alpha = score

                if alpha >= beta:
                    break

            return best_score

        stand_pat = module.evaluate(board)

        if stand_pat >= beta:
            return stand_pat

        if stand_pat > alpha:
            alpha = stand_pat

        tactical = []
        quiet_checks = []

        for move in moves:
            if board.is_capture(move) or move.promotion is not None:
                tactical.append(move)
            elif check_budget > 0 and board.gives_check(move):
                quiet_checks.append(move)

        tactical.sort(
            key=lambda move: qmove_score(board, move),
            reverse=True,
        )
        quiet_checks.sort(
            key=lambda move: qmove_score(board, move),
            reverse=True,
        )

        # Captures/promotions retain the check budget. They were already part of
        # submitted V1 qsearch, so this path is deliberately unchanged in scope.
        for move in tactical:
            board.push(move)
            try:
                score = -quiescence(
                    board,
                    -beta,
                    -alpha,
                    ply + 1,
                    check_budget,
                )
            finally:
                board.pop()

            if score >= beta:
                return score

            if score > alpha:
                alpha = score

        # New experiment: allow one quiet checking move at the frontier.
        if check_budget > 0:
            for move in quiet_checks:
                board.push(move)
                try:
                    score = -quiescence(
                        board,
                        -beta,
                        -alpha,
                        ply + 1,
                        check_budget - 1,
                    )
                finally:
                    board.pop()

                if score >= beta:
                    return score

                if score > alpha:
                    alpha = score

        return alpha

    module.quiescence = quiescence


def load_fens(path: Path, max_fens: int) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"FEN file not found: {path}")

    fens = []
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


def sf_score(info: dict, colour: chess.Color) -> int:
    value = info["score"].pov(colour).score(mate_score=2000)
    if value is None:
        raise RuntimeError("Could not convert Stockfish score")
    return int(value)


def classify(loss: int) -> str:
    if loss >= 300:
        return "blunder"
    if loss >= 150:
        return "major"
    if loss >= 75:
        return "mistake"
    if loss >= 30:
        return "inaccuracy"
    return "ok"


def reset_search_state(module) -> None:
    module.TT.clear()
    module.EVAL_CACHE.clear()
    module.KILLERS.clear()
    module.HISTORY.clear()
    module.NODES = 0


def engine_move(module, fen: str, time_left_ms: int) -> tuple[str, int]:
    reset_search_state(module)
    move = module.get_move(fen, time_left_ms)
    nodes = module.NODES
    return move, nodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("/tmp/learned_v1"),
    )
    parser.add_argument(
        "--fens",
        type=Path,
        default=Path("training/data/test_fens_60.txt"),
    )
    parser.add_argument("--max-fens", type=int, default=60)
    parser.add_argument("--time-left-ms", type=int, default=5000)
    parser.add_argument(
        "--stockfish",
        type=Path,
        default=Path("/usr/games/stockfish"),
    )
    parser.add_argument("--stockfish-depth", type=int, default=13)
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("qsearch_v2_ab.csv"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("qsearch_v2_ab.txt"),
    )
    args = parser.parse_args()

    agent_path = args.baseline / "agent.py"
    if not agent_path.is_file():
        raise SystemExit(f"Submitted V1 agent not found: {agent_path}")
    if not args.stockfish.is_file():
        raise SystemExit(f"Stockfish not found: {args.stockfish}")

    fens = load_fens(args.fens, args.max_fens)

    baseline = load_agent("submitted_v1_ab", agent_path)
    candidate = load_agent("qsearch_v2_ab", agent_path)
    install_qsearch_v2(candidate)

    print(
        f"Loaded exact submitted V1 twice. Testing {len(fens)} positions.",
        flush=True,
    )
    print(
        "Only experimental difference: one quiet-check ply in quiescence.",
        flush=True,
    )

    sf = chess.engine.SimpleEngine.popen_uci(str(args.stockfish))
    options = {}
    if "Threads" in sf.options:
        options["Threads"] = 1
    if "Hash" in sf.options:
        options["Hash"] = 256
    if "UCI_LimitStrength" in sf.options:
        options["UCI_LimitStrength"] = False
    if options:
        sf.configure(options)

    rows = []
    losses = {"v1": [], "qsearch_v2": []}
    nodes = {"v1": [], "qsearch_v2": []}
    changed = 0
    candidate_better = 0
    baseline_better = 0
    equal = 0

    try:
        for index, fen in enumerate(fens, start=1):
            board = chess.Board(fen)
            mover = board.turn

            v1_move_text, v1_nodes = engine_move(
                baseline,
                fen,
                args.time_left_ms,
            )
            q_move_text, q_nodes = engine_move(
                candidate,
                fen,
                args.time_left_ms,
            )

            v1_move = chess.Move.from_uci(v1_move_text)
            q_move = chess.Move.from_uci(q_move_text)

            if v1_move not in board.legal_moves:
                raise RuntimeError(f"V1 illegal move on FEN {index}: {v1_move}")
            if q_move not in board.legal_moves:
                raise RuntimeError(f"QSearch V2 illegal move on FEN {index}: {q_move}")

            best_info = sf.analyse(
                board,
                chess.engine.Limit(depth=args.stockfish_depth),
            )
            best_score = sf_score(best_info, mover)
            best_pv = best_info.get("pv", [])
            best_move = best_pv[0] if best_pv else None

            move_results = {}

            for label, move, node_count in (
                ("v1", v1_move, v1_nodes),
                ("qsearch_v2", q_move, q_nodes),
            ):
                after = board.copy(stack=False)
                after.push(move)
                info = sf.analyse(
                    after,
                    chess.engine.Limit(depth=args.stockfish_depth),
                )
                played_score = sf_score(info, mover)
                cp_loss = min(4000, max(0, best_score - played_score))

                losses[label].append(cp_loss)
                nodes[label].append(node_count)
                move_results[label] = (played_score, cp_loss)

            v1_loss = move_results["v1"][1]
            q_loss = move_results["qsearch_v2"][1]

            if v1_move != q_move:
                changed += 1

            if q_loss < v1_loss:
                candidate_better += 1
            elif v1_loss < q_loss:
                baseline_better += 1
            else:
                equal += 1

            rows.append(
                {
                    "fen_index": index,
                    "fen": fen,
                    "stockfish_best": "" if best_move is None else best_move.uci(),
                    "best_eval_cp": best_score,
                    "v1_move": v1_move.uci(),
                    "v1_cp_loss": v1_loss,
                    "v1_class": classify(v1_loss),
                    "v1_nodes": v1_nodes,
                    "qsearch_v2_move": q_move.uci(),
                    "qsearch_v2_cp_loss": q_loss,
                    "qsearch_v2_class": classify(q_loss),
                    "qsearch_v2_nodes": q_nodes,
                }
            )

            print(
                f"{index:>3}/{len(fens)} "
                f"V1 {v1_move.uci():>5} {v1_loss:>4}cp | "
                f"Q2 {q_move.uci():>5} {q_loss:>4}cp",
                flush=True,
            )

    finally:
        sf.quit()

    with args.csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def stats(label: str) -> dict[str, float]:
        values = losses[label]
        return {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "lt30": sum(x < 30 for x in values) / len(values),
            "ge75": sum(x >= 75 for x in values) / len(values),
            "ge300": sum(x >= 300 for x in values) / len(values),
            "nodes": statistics.fmean(nodes[label]),
        }

    a = stats("v1")
    b = stats("qsearch_v2")

    lines = [
        "",
        "=" * 78,
        "V1 QUIESCENCE A/B",
        "=" * 78,
        f"Positions: {len(fens)}",
        f"Candidate clock input: {args.time_left_ms} ms",
        f"Stockfish analysis depth: {args.stockfish_depth}",
        "",
        f"{'engine':<14} {'mean':>8} {'median':>8} {'<30cp':>8} "
        f"{'>=75':>8} {'>=300':>8} {'avg nodes':>12}",
        "-" * 78,
        (
            f"{'submitted_v1':<14} {a['mean']:>7.1f} {a['median']:>7.1f} "
            f"{100*a['lt30']:>7.1f}% {100*a['ge75']:>7.1f}% "
            f"{100*a['ge300']:>7.1f}% {a['nodes']:>12,.0f}"
        ),
        (
            f"{'qsearch_v2':<14} {b['mean']:>7.1f} {b['median']:>7.1f} "
            f"{100*b['lt30']:>7.1f}% {100*b['ge75']:>7.1f}% "
            f"{100*b['ge300']:>7.1f}% {b['nodes']:>12,.0f}"
        ),
        "",
        f"Different root moves: {changed}/{len(fens)}",
        f"QSearch V2 lower Stockfish loss: {candidate_better}",
        f"Submitted V1 lower Stockfish loss: {baseline_better}",
        f"Equal loss: {equal}",
        "",
        "This experiment changes quiescence only.",
        f"CSV: {args.csv_out}",
    ]

    text = "\n".join(lines)
    print(text)
    args.summary_out.write_text(text + "\n", encoding="utf-8")
    print(f"Summary saved to: {args.summary_out}")


if __name__ == "__main__":
    main()

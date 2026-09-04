"""A/B test one isolated search optimization: faster V1 quiescence.

Both engines load the exact submitted V1 from /tmp/learned_v1.

The candidate keeps V1 qsearch semantics:
    * full legal evasions when in check
    * stand pat when not in check
    * captures and promotions only
    * the same V1 move ordering

The only intended change is move-generation efficiency:
    * do not build a full legal-move list at every quiet qsearch node
    * generate legal captures directly
    * generate non-capture promotions only when a pawn can promote
    * test for stalemate with a single legal-move probe when necessary

No evaluator, pruning, LMR, null move, aspiration, time-manager, material,
or repetition changes are made.

Run from the repository root:

    python -m training.test_qsearch_fast \
        --fens training/data/test_fens_60.txt \
        --max-fens 60 \
        --time-left-ms 5000

For a smaller tournament-clock confirmation:

    python -m training.test_qsearch_fast \
        --fens training/data/test_fens_60.txt \
        --max-fens 20 \
        --time-left-ms 120000 \
        --csv-out qsearch_fast_tournament.csv \
        --summary-out qsearch_fast_tournament.txt
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
import sys
import time
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


def install_fast_qsearch(module) -> None:
    """Replace only module.quiescence with a semantically equivalent version."""

    def quiescence(board, alpha, beta, ply):
        module.NODES += 1
        module.check_time()

        if board.halfmove_clock >= 100:
            return 0

        in_check = board.is_check()

        if in_check:
            # In check, stand pat is illegal. Exactly as in submitted V1,
            # search every legal evasion and detect checkmate if none exist.
            moves = list(board.legal_moves)

            if not moves:
                return -module.MATE + ply

            module.order_move_list(board, moves, None, ply)
            best_score = -module.INF

            for move in moves:
                board.push(move)
                try:
                    score = -quiescence(
                        board,
                        -beta,
                        -alpha,
                        ply + 1,
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

        # Quiet node. Generate captures directly instead of materialising every
        # legal move merely to throw most of them away.
        tactical_moves = list(board.generate_legal_captures())

        # Non-capture promotions are also part of submitted V1 qsearch.
        promotion_rank = 6 if board.turn == chess.WHITE else 1
        pawns = board.pieces(chess.PAWN, board.turn)

        if any(chess.square_rank(square) == promotion_rank for square in pawns):
            for move in board.legal_moves:
                if move.promotion is not None and not board.is_capture(move):
                    tactical_moves.append(move)

        # If there is no tactical move, verify that the position is not
        # stalemate. `any()` normally needs only the first legal move rather
        # than constructing the entire legal move list.
        if not tactical_moves:
            if not any(board.generate_legal_moves()):
                return 0

            stand_pat = module.evaluate(board)

            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat

            return alpha

        # A tactical move proves the position is not stalemate, so no full
        # legal-move generation is required.
        stand_pat = module.evaluate(board)

        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

        module.order_move_list(
            board,
            tactical_moves,
            None,
            ply,
        )

        for move in tactical_moves:
            board.push(move)
            try:
                score = -quiescence(
                    board,
                    -beta,
                    -alpha,
                    ply + 1,
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


def engine_move(
    module,
    fen: str,
    time_left_ms: int,
) -> tuple[str, int, float]:
    reset_search_state(module)

    start = time.perf_counter()
    move = module.get_move(fen, time_left_ms)
    elapsed = time.perf_counter() - start

    return move, module.NODES, elapsed


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
        default=Path("qsearch_fast_ab.csv"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("qsearch_fast_ab.txt"),
    )

    args = parser.parse_args()

    agent_path = args.baseline / "agent.py"

    if not agent_path.is_file():
        raise SystemExit(f"Submitted V1 agent not found: {agent_path}")

    if not args.stockfish.is_file():
        raise SystemExit(f"Stockfish not found: {args.stockfish}")

    fens = load_fens(
        args.fens,
        args.max_fens,
    )

    baseline = load_agent(
        "submitted_v1_fastq_ab",
        agent_path,
    )
    candidate = load_agent(
        "fast_qsearch_ab",
        agent_path,
    )

    install_fast_qsearch(candidate)

    print(
        f"Loaded exact submitted V1 twice. Testing {len(fens)} positions.",
        flush=True,
    )
    print(
        "Only experimental difference: qsearch move-generation efficiency.",
        flush=True,
    )

    sf = chess.engine.SimpleEngine.popen_uci(
        str(args.stockfish)
    )

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

    losses = {
        "v1": [],
        "fast": [],
    }
    nodes = {
        "v1": [],
        "fast": [],
    }
    walls = {
        "v1": [],
        "fast": [],
    }

    changed = 0
    fast_better = 0
    v1_better = 0
    equal = 0

    sf_after_cache: dict[tuple[str, str], int] = {}

    try:
        for index, fen in enumerate(fens, start=1):
            board = chess.Board(fen)
            mover = board.turn

            v1_text, v1_nodes, v1_wall = engine_move(
                baseline,
                fen,
                args.time_left_ms,
            )
            fast_text, fast_nodes, fast_wall = engine_move(
                candidate,
                fen,
                args.time_left_ms,
            )

            v1_move = chess.Move.from_uci(v1_text)
            fast_move = chess.Move.from_uci(fast_text)

            if v1_move not in board.legal_moves:
                raise RuntimeError(
                    f"V1 illegal move on FEN {index}: {v1_move}"
                )

            if fast_move not in board.legal_moves:
                raise RuntimeError(
                    f"Fast qsearch illegal move on FEN {index}: {fast_move}"
                )

            best_info = sf.analyse(
                board,
                chess.engine.Limit(
                    depth=args.stockfish_depth
                ),
            )
            best_score = sf_score(
                best_info,
                mover,
            )
            best_pv = best_info.get("pv", [])
            best_move = best_pv[0] if best_pv else None

            results = {}

            for label, move, node_count, wall in (
                ("v1", v1_move, v1_nodes, v1_wall),
                ("fast", fast_move, fast_nodes, fast_wall),
            ):
                cache_key = (
                    fen,
                    move.uci(),
                )

                played_score = sf_after_cache.get(
                    cache_key
                )

                if played_score is None:
                    after = board.copy(stack=False)
                    after.push(move)

                    info = sf.analyse(
                        after,
                        chess.engine.Limit(
                            depth=args.stockfish_depth
                        ),
                    )
                    played_score = sf_score(
                        info,
                        mover,
                    )
                    sf_after_cache[cache_key] = (
                        played_score
                    )

                cp_loss = min(
                    4000,
                    max(
                        0,
                        best_score - played_score,
                    ),
                )

                losses[label].append(cp_loss)
                nodes[label].append(node_count)
                walls[label].append(wall)
                results[label] = cp_loss

            v1_loss = results["v1"]
            fast_loss = results["fast"]

            if v1_move != fast_move:
                changed += 1

            if fast_loss < v1_loss:
                fast_better += 1
            elif v1_loss < fast_loss:
                v1_better += 1
            else:
                equal += 1

            rows.append(
                {
                    "fen_index": index,
                    "fen": fen,
                    "stockfish_best": (
                        ""
                        if best_move is None
                        else best_move.uci()
                    ),
                    "best_eval_cp": best_score,
                    "v1_move": v1_move.uci(),
                    "v1_cp_loss": v1_loss,
                    "v1_class": classify(v1_loss),
                    "v1_nodes": v1_nodes,
                    "v1_wall_s": v1_wall,
                    "fast_move": fast_move.uci(),
                    "fast_cp_loss": fast_loss,
                    "fast_class": classify(fast_loss),
                    "fast_nodes": fast_nodes,
                    "fast_wall_s": fast_wall,
                }
            )

            print(
                f"{index:>3}/{len(fens)} "
                f"V1 {v1_move.uci():>5} {v1_loss:>4}cp "
                f"{v1_nodes:>7,}n | "
                f"FAST {fast_move.uci():>5} {fast_loss:>4}cp "
                f"{fast_nodes:>7,}n",
                flush=True,
            )

    finally:
        sf.quit()

    with args.csv_out.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    def stats(label: str) -> dict[str, float]:
        values = losses[label]

        return {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "lt30": (
                sum(x < 30 for x in values)
                / len(values)
            ),
            "ge75": (
                sum(x >= 75 for x in values)
                / len(values)
            ),
            "ge300": (
                sum(x >= 300 for x in values)
                / len(values)
            ),
            "nodes": statistics.fmean(nodes[label]),
            "wall": statistics.fmean(walls[label]),
        }

    a = stats("v1")
    b = stats("fast")

    node_gain = (
        (b["nodes"] / a["nodes"] - 1.0) * 100.0
        if a["nodes"]
        else 0.0
    )

    lines = [
        "",
        "=" * 94,
        "V1 FAST-QUIESCENCE A/B",
        "=" * 94,
        f"Positions: {len(fens)}",
        f"Clock input: {args.time_left_ms} ms",
        f"Stockfish depth: {args.stockfish_depth}",
        "",
        (
            f"{'engine':<14} "
            f"{'mean':>8} "
            f"{'median':>8} "
            f"{'<30cp':>8} "
            f"{'>=75':>8} "
            f"{'>=300':>8} "
            f"{'avg nodes':>12} "
            f"{'avg wall':>10}"
        ),
        "-" * 94,
        (
            f"{'submitted_v1':<14} "
            f"{a['mean']:>7.1f} "
            f"{a['median']:>7.1f} "
            f"{100*a['lt30']:>7.1f}% "
            f"{100*a['ge75']:>7.1f}% "
            f"{100*a['ge300']:>7.1f}% "
            f"{a['nodes']:>12,.0f} "
            f"{a['wall']:>9.3f}s"
        ),
        (
            f"{'fast_qsearch':<14} "
            f"{b['mean']:>7.1f} "
            f"{b['median']:>7.1f} "
            f"{100*b['lt30']:>7.1f}% "
            f"{100*b['ge75']:>7.1f}% "
            f"{100*b['ge300']:>7.1f}% "
            f"{b['nodes']:>12,.0f} "
            f"{b['wall']:>9.3f}s"
        ),
        "",
        f"Average node change: {node_gain:+.1f}%",
        f"Different root moves: {changed}/{len(fens)}",
        f"Fast qsearch lower Stockfish loss: {fast_better}",
        f"Submitted V1 lower Stockfish loss: {v1_better}",
        f"Equal loss: {equal}",
        "",
        "This experiment changes qsearch implementation efficiency only.",
        f"CSV: {args.csv_out}",
    ]

    text = "\n".join(lines)

    print(text)

    args.summary_out.write_text(
        text + "\n",
        encoding="utf-8",
    )

    print(
        f"Summary saved to: {args.summary_out}"
    )


if __name__ == "__main__":
    main()

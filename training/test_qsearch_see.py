"""A/B test one isolated search change: conservative SEE inside V1 quiescence.

Both engines load the exact submitted V1 from /tmp/learned_v1.
The candidate changes ONLY quiescence capture handling:

1. Compute a local static-exchange score (SEE) for each capture.
2. Order qsearch captures by SEE.
3. Skip only clearly losing non-checking, non-promotion captures
   with SEE < --see-prune-cp (default: -100 cp).

No changes to:
    evaluator
    material weighting
    quiet checks
    LMR
    null-move pruning
    futility pruning
    aspiration windows
    time management
    root search

Run from repository root:

    python -m training.test_qsearch_see \
        --fens training/data/test_fens_60.txt \
        --max-fens 60 \
        --time-left-ms 5000

Outputs:
    qsearch_see_ab.csv
    qsearch_see_ab.txt
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


def captured_value(module, board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return module.PIECE_VALUE[chess.PAWN]

    victim = board.piece_at(move.to_square)
    if victim is None:
        return 0

    return module.PIECE_VALUE.get(victim.piece_type, 0)


def promotion_gain(module, move: chess.Move) -> int:
    if move.promotion is None:
        return 0

    return (
        module.PIECE_VALUE.get(move.promotion, 0)
        - module.PIECE_VALUE[chess.PAWN]
    )


def exchange_reply_gain(module, board: chess.Board, target: chess.Square) -> int:
    """Best optional recapture sequence for side to move on one square.

    This is deliberately local. It considers only legal captures onto `target`.
    Because legal moves are used, pins and king-safety constraints are respected.
    The side to move may decline the exchange, hence the floor at zero.
    """
    piece_on_target = board.piece_at(target)
    if piece_on_target is None:
        return 0

    best = 0

    captures = [
        move
        for move in board.legal_moves
        if board.is_capture(move) and move.to_square == target
    ]

    for move in captures:
        gain_now = captured_value(module, board, move)
        gain_now += promotion_gain(module, move)

        board.push(move)
        try:
            continuation = exchange_reply_gain(module, board, target)
        finally:
            board.pop()

        net = gain_now - continuation
        if net > best:
            best = net

    return best


def see_capture(module, board: chess.Board, move: chess.Move) -> int:
    """Centipawn result of the local capture/recapture sequence."""
    gain_now = captured_value(module, board, move)
    gain_now += promotion_gain(module, move)

    target = move.to_square

    board.push(move)
    try:
        reply = exchange_reply_gain(module, board, target)
    finally:
        board.pop()

    return gain_now - reply


def install_see_qsearch(module, prune_cp: int) -> None:
    """Replace only module.quiescence."""

    def quiescence(board, alpha, beta, ply):
        module.NODES += 1
        module.check_time()

        moves = list(board.legal_moves)

        if not moves:
            if board.is_check():
                return -module.MATE + ply
            return 0

        if board.halfmove_clock >= 100:
            return 0

        # Submitted V1 behavior in check: no stand-pat, search all evasions.
        if board.is_check():
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

        stand_pat = module.evaluate(board)

        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

        tactical = []

        for move in moves:
            if not (board.is_capture(move) or move.promotion is not None):
                continue

            gives_check = board.gives_check(move)

            if board.is_capture(move):
                see = see_capture(module, board, move)
            else:
                # Non-capture promotion.
                see = promotion_gain(module, move)

            # Be conservative. Never prune promotions or checking captures.
            if (
                board.is_capture(move)
                and move.promotion is None
                and not gives_check
                and see < -prune_cp
            ):
                continue

            tactical.append((see, move))

        # Good exchanges first. This improves alpha-beta efficiency even for
        # captures that are retained.
        tactical.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        for _see, move in tactical:
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
    parser.add_argument(
        "--max-fens",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--time-left-ms",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--stockfish",
        type=Path,
        default=Path("/usr/games/stockfish"),
    )
    parser.add_argument(
        "--stockfish-depth",
        type=int,
        default=13,
    )
    parser.add_argument(
        "--see-prune-cp",
        type=int,
        default=100,
        help="Prune qsearch captures only when SEE < -this value.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("qsearch_see_ab.csv"),
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("qsearch_see_ab.txt"),
    )

    args = parser.parse_args()

    agent_path = args.baseline / "agent.py"

    if not agent_path.is_file():
        raise SystemExit(f"Submitted V1 agent not found: {agent_path}")

    if not args.stockfish.is_file():
        raise SystemExit(f"Stockfish not found: {args.stockfish}")

    fens = load_fens(args.fens, args.max_fens)

    baseline = load_agent(
        "submitted_v1_see_ab",
        agent_path,
    )
    candidate = load_agent(
        "see_qsearch_ab",
        agent_path,
    )

    install_see_qsearch(
        candidate,
        args.see_prune_cp,
    )

    print(
        f"Loaded exact submitted V1 twice. Testing {len(fens)} positions.",
        flush=True,
    )
    print(
        "Only experimental difference: conservative SEE in quiescence.",
        flush=True,
    )
    print(
        f"Prune threshold: SEE < -{args.see_prune_cp} cp",
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
        "see": [],
    }

    nodes = {
        "v1": [],
        "see": [],
    }

    changed = 0
    see_better = 0
    v1_better = 0
    equal = 0

    # Ensure identical Stockfish scoring when both engines choose the same move.
    sf_after_cache: dict[tuple[str, str], int] = {}

    try:
        for index, fen in enumerate(fens, start=1):
            board = chess.Board(fen)
            mover = board.turn

            v1_move_text, v1_nodes = engine_move(
                baseline,
                fen,
                args.time_left_ms,
            )

            see_move_text, see_nodes = engine_move(
                candidate,
                fen,
                args.time_left_ms,
            )

            v1_move = chess.Move.from_uci(v1_move_text)
            see_move = chess.Move.from_uci(see_move_text)

            if v1_move not in board.legal_moves:
                raise RuntimeError(
                    f"V1 illegal move on FEN {index}: {v1_move}"
                )

            if see_move not in board.legal_moves:
                raise RuntimeError(
                    f"SEE candidate illegal move on FEN {index}: {see_move}"
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

            move_results = {}

            for label, move, node_count in (
                ("v1", v1_move, v1_nodes),
                ("see", see_move, see_nodes),
            ):
                key = (
                    fen,
                    move.uci(),
                )

                played_score = sf_after_cache.get(key)

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

                    sf_after_cache[key] = played_score

                cp_loss = min(
                    4000,
                    max(
                        0,
                        best_score - played_score,
                    ),
                )

                losses[label].append(cp_loss)
                nodes[label].append(node_count)
                move_results[label] = (
                    played_score,
                    cp_loss,
                )

            v1_loss = move_results["v1"][1]
            see_loss = move_results["see"][1]

            if v1_move != see_move:
                changed += 1

            if see_loss < v1_loss:
                see_better += 1
            elif v1_loss < see_loss:
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
                    "see_move": see_move.uci(),
                    "see_cp_loss": see_loss,
                    "see_class": classify(see_loss),
                    "see_nodes": see_nodes,
                }
            )

            print(
                f"{index:>3}/{len(fens)} "
                f"V1 {v1_move.uci():>5} {v1_loss:>4}cp | "
                f"SEE {see_move.uci():>5} {see_loss:>4}cp",
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
        }

    a = stats("v1")
    b = stats("see")

    lines = [
        "",
        "=" * 82,
        "V1 SEE-QUIESCENCE A/B",
        "=" * 82,
        f"Positions: {len(fens)}",
        f"Candidate clock input: {args.time_left_ms} ms",
        f"Stockfish analysis depth: {args.stockfish_depth}",
        f"SEE prune threshold: < -{args.see_prune_cp} cp",
        "",
        (
            f"{'engine':<14} "
            f"{'mean':>8} "
            f"{'median':>8} "
            f"{'<30cp':>8} "
            f"{'>=75':>8} "
            f"{'>=300':>8} "
            f"{'avg nodes':>12}"
        ),
        "-" * 82,
        (
            f"{'submitted_v1':<14} "
            f"{a['mean']:>7.1f} "
            f"{a['median']:>7.1f} "
            f"{100*a['lt30']:>7.1f}% "
            f"{100*a['ge75']:>7.1f}% "
            f"{100*a['ge300']:>7.1f}% "
            f"{a['nodes']:>12,.0f}"
        ),
        (
            f"{'see_qsearch':<14} "
            f"{b['mean']:>7.1f} "
            f"{b['median']:>7.1f} "
            f"{100*b['lt30']:>7.1f}% "
            f"{100*b['ge75']:>7.1f}% "
            f"{100*b['ge300']:>7.1f}% "
            f"{b['nodes']:>12,.0f}"
        ),
        "",
        f"Different root moves: {changed}/{len(fens)}",
        f"SEE candidate lower Stockfish loss: {see_better}",
        f"Submitted V1 lower Stockfish loss: {v1_better}",
        f"Equal loss: {equal}",
        "",
        "This experiment changes quiescence capture handling only.",
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

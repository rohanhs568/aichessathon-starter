"""Deterministic tests for Chessathon repetition/draw handling, V3.

This version tests the *invariant we actually care about*:
after every simulated real move, the persistent protocol-side occurrence
counter must equal an independently reconstructed count from the complete
python-chess move stack.

It does not assume that a FEN itself stores history. The same agent process
persists across tournament calls, so the agent records its own move and infers
the one legal opponent reply that reaches the next incoming FEN.

The test covers:
  1. python-chess ground-truth threefold semantics;
  2. repetition-key semantics (clocks excluded, legal EP/castling included);
  3. persistent FEN-only protocol synchronization after every ply;
  4. exact third-occurrence draw scoring;
  5. optional first-cycle search-tree scoring.

On failure it prints detailed state/count differences instead of a bare assert.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import os
from contextlib import redirect_stdout
from pathlib import Path

import chess


def load_lab(path: Path, weights: Path | None):
    if weights is not None:
        os.environ["SEARCHLAB_MODEL_PATH"] = str(weights.resolve())
    spec = importlib.util.spec_from_file_location("agent_searchlab_rep_v3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    with redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def knight_cycle_board(cycles: int):
    board = chess.Board()
    cycle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    for _ in range(cycles):
        for uci in cycle:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise AssertionError(f"illegal test move {uci} at {board.fen()}")
            board.push(move)
    return board


def persist_our_known_move(lab, root: chess.Board, move_uci: str):
    """Mirror the history update done at the end of lab.get_move()."""
    move = chess.Move.from_uci(move_uci)
    if move not in root.legal_moves:
        raise AssertionError(f"illegal own test move {move_uci}")

    after = root.copy(stack=False)
    after.push(move)
    lab._GAME_BOARD = after.copy(stack=False)
    lab._record_game_position(lab._GAME_BOARD)
    return after


def format_counts(counts):
    rows = sorted(((repr(k), v) for k, v in counts.items()), key=lambda x: x[0])
    return "\n".join(f"    {v} x {k}" for k, v in rows)


def assert_counts_equal(lab, actual: chess.Board, label: str):
    expected = lab._build_game_counts(actual)
    observed = lab._game_counts_snapshot()

    if observed != expected:
        missing = {k: v for k, v in expected.items() if observed.get(k) != v}
        extra = {k: v for k, v in observed.items() if expected.get(k) != v}
        raise AssertionError(
            f"protocol count mismatch at {label}\n"
            f"actual FEN: {actual.fen()}\n"
            f"GAME_BOARD: {None if lab._GAME_BOARD is None else lab._GAME_BOARD.fen()}\n"
            f"expected total={sum(expected.values())}, observed total={sum(observed.values())}\n"
            f"expected differing keys:\n{format_counts(missing) or '    <none>'}\n"
            f"observed differing keys:\n{format_counts(extra) or '    <none>'}"
        )


def simulate_protocol_history(lab):
    """Pretend the engine is White and receives one FEN every two plies."""
    lab.reset_searchlab_state(reset_game=True)

    actual = chess.Board()
    synced = lab._sync_game_board(actual.fen())
    if lab._protocol_state(synced) != lab._protocol_state(actual):
        raise AssertionError("initial protocol sync failed")
    assert_counts_equal(lab, actual, "initial seed")

    start_key = lab._rep_key(actual)

    white_moves = ["g1f3", "f3g1", "g1f3", "f3g1"]
    black_moves = ["g8f6", "f6g8", "g8f6", "f6g8"]
    trace = []

    for pair, (white_uci, black_uci) in enumerate(
        zip(white_moves, black_moves), start=1
    ):
        # Duplicate request for the current root must be idempotent.
        before = lab._game_counts_snapshot()
        root = lab._sync_game_board(actual.fen())
        if lab._protocol_state(root) != lab._protocol_state(actual):
            raise AssertionError(f"root sync failed before pair {pair}")
        if lab._game_counts_snapshot() != before:
            raise AssertionError(f"duplicate root sync double-counted at pair {pair}")

        # Our move is known exactly and is recorded immediately.
        persist_our_known_move(lab, root, white_uci)
        own_move = chess.Move.from_uci(white_uci)
        actual.push(own_move)

        if lab._protocol_state(lab._GAME_BOARD) != lab._protocol_state(actual):
            raise AssertionError(f"own-move persistence diverged at pair {pair}")

        assert_counts_equal(lab, actual, f"pair {pair}, after our move")

        # Opponent moves outside our process. Next incoming FEN must allow the
        # agent to infer exactly that one move and record exactly one position.
        opp_move = chess.Move.from_uci(black_uci)
        if opp_move not in actual.legal_moves:
            raise AssertionError(f"illegal opponent test move {black_uci}")
        actual.push(opp_move)

        synced = lab._sync_game_board(actual.fen())
        if lab._protocol_state(synced) != lab._protocol_state(actual):
            raise AssertionError(f"opponent-reply sync diverged at pair {pair}")

        assert_counts_equal(lab, actual, f"pair {pair}, after opponent move")

        trace.append(
            (
                pair,
                len(actual.move_stack),
                lab._GAME_COUNTS.get(start_key, 0),
                sum(lab._GAME_COUNTS.values()),
            )
        )

    return actual, synced, dict(lab._GAME_COUNTS), trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent-lab",
        type=Path,
        default=Path("training/searchlab_agent.py"),
    )
    parser.add_argument("--weights", type=Path, default=Path("weights/v1.npz"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("repetition_design_test.txt"),
    )
    args = parser.parse_args()

    lab = load_lab(args.agent_lab, args.weights)
    lines = []

    # 1. Ground truth.
    once = knight_cycle_board(1)
    twice = knight_cycle_board(2)

    start = chess.Board()
    start_key = lab._rep_key(start)
    once_counts = lab._build_game_counts(once)
    twice_counts = lab._build_game_counts(twice)

    if once_counts.get(start_key) != 2:
        raise AssertionError(
            f"one-cycle expected two start occurrences, got "
            f"{once_counts.get(start_key)}"
        )
    if twice_counts.get(start_key) != 3:
        raise AssertionError(
            f"two-cycle expected three start occurrences, got "
            f"{twice_counts.get(start_key)}"
        )
    if once.can_claim_threefold_repetition():
        raise AssertionError("python-chess incorrectly claims threefold after one cycle")
    if not twice.can_claim_threefold_repetition():
        raise AssertionError("python-chess did not claim threefold after two cycles")
    outcome = twice.outcome(claim_draw=True)
    if outcome is None or outcome.winner is not None:
        raise AssertionError(f"expected draw outcome, got {outcome}")

    lines += [
        "RULE SEMANTICS: PASS",
        f"starting position occurrences after one cycle: {once_counts[start_key]}",
        f"starting position occurrences after two cycles: {twice_counts[start_key]}",
        f"python-chess claim after two cycles: {twice.can_claim_threefold_repetition()}",
        f"python-chess outcome: {outcome}",
        "",
    ]

    # 2. Verify clocks are excluded from repetition identity.
    clock_variant = chess.Board()
    clock_variant.halfmove_clock = 37
    clock_variant.fullmove_number = 19
    if lab._rep_key(clock_variant) != start_key:
        raise AssertionError("repetition key incorrectly depends on FEN clocks")

    lines += [
        "REPETITION KEY CLOCK-INDEPENDENCE: PASS",
        "halfmove/fullmove counters do not alter repetition identity",
        "",
    ]

    # 3. FEN-only protocol reconstruction.
    actual, synced, protocol_counts, trace = simulate_protocol_history(lab)

    if protocol_counts.get(start_key) != 3:
        raise AssertionError(
            f"protocol reconstruction expected 3 starting-position occurrences, "
            f"got {protocol_counts.get(start_key)}\n"
            f"trace={trace}"
        )
    if sum(protocol_counts.values()) != 9:
        raise AssertionError(
            f"expected 9 actual position occurrences, got "
            f"{sum(protocol_counts.values())}; trace={trace}"
        )

    lines += [
        "FEN-ONLY PROTOCOL HISTORY RECONSTRUCTION: PASS",
        f"actual plies simulated: {len(actual.move_stack)}",
        f"starting position occurrences reconstructed: {protocol_counts[start_key]}",
        f"total actual position occurrences recorded: {sum(protocol_counts.values())}",
    ]
    for pair, plies, start_occ, total_occ in trace:
        lines.append(
            f"  pair={pair} plies={plies} "
            f"start_occurrences={start_occ} total_occurrences={total_occ}"
        )
    lines.append("")

    # 4. Exact policy.
    lab.set_searchlab_variant("repetition_exact")
    lab._GAME_COUNTS = dict(protocol_counts)
    lab._PATH_COUNTS = {}
    if not lab._is_repetition_draw(synced):
        raise AssertionError("exact policy failed to score true third occurrence as draw")

    one_cycle = knight_cycle_board(1)
    lab.reset_searchlab_state(reset_game=True)
    lab.set_searchlab_variant("repetition_exact")
    lab._GAME_COUNTS = lab._build_game_counts(one_cycle)
    lab._PATH_COUNTS = {}
    if lab._is_repetition_draw(one_cycle):
        raise AssertionError("exact policy incorrectly scores second occurrence as draw")

    lines += [
        "EXACT THREEFOLD POLICY: PASS",
        "third occurrence returns draw score 0",
        "EXACT TWO-OCCURRENCE NON-DRAW: PASS",
        "second occurrence is not incorrectly scored as threefold",
        "",
    ]

    # 5. Optional first-cycle policy.
    lab.reset_searchlab_state(reset_game=True)
    root = chess.Board()
    lab._GAME_COUNTS = {lab._rep_key(root): 1}
    lab._PATH_COUNTS = {}
    lab.set_searchlab_variant("repetition_cycle")

    path_board = root.copy(stack=False)
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]:
        move = chess.Move.from_uci(uci)
        if move not in path_board.legal_moves:
            raise AssertionError(f"illegal path move {uci}")
        lab._rep_push(path_board, move)

    if lab._rep_key(path_board) != lab._rep_key(root):
        raise AssertionError("cycle did not return to repetition-identical root")
    if not lab._is_repetition_draw(path_board):
        raise AssertionError("first-cycle search policy failed")

    lines += [
        "SEARCH-TREE FIRST-CYCLE POLICY: PASS",
        "second occurrence inside current search is treated as 0",
        "",
        "IMPORTANT",
        "---------",
        "Exact threefold and first-cycle scoring remain separate policies.",
        "Exact threefold tracking is a correctness requirement.",
        "First-cycle scoring is a search heuristic and needs A/B testing.",
    ]

    args.output.write_text("\n".join(lines) + "\n")
    print(args.output.read_text())


if __name__ == "__main__":
    main()

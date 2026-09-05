"""Deterministic capture-mechanics sanity tests for the production candidate.

These are deliberately simple positions. They are not a strength benchmark.
They answer one narrow question: can the search/evaluator correctly prefer an
immediately winning capture when there is essentially no strategic ambiguity?

Run from repository root after installing candidate agent.py + weights/v1.npz.
"""
from __future__ import annotations

import math

import chess
import agent


def make_board(pieces, turn=chess.WHITE, ep_square=None):
    board = chess.Board(None)
    for square_name, symbol in pieces:
        board.set_piece_at(
            chess.parse_square(square_name),
            chess.Piece.from_symbol(symbol),
        )
    board.turn = turn
    board.castling_rights = chess.BB_EMPTY
    board.ep_square = None if ep_square is None else chess.parse_square(ep_square)
    board.halfmove_clock = 0
    board.fullmove_number = 1
    board.clear_stack()
    if not board.is_valid():
        raise RuntimeError(f"invalid sanity position: {board.fen()}")
    return board


def reset_for_root(board):
    agent.TT.clear()
    agent.EVAL_CACHE.clear()
    agent.KILLERS.clear()
    agent.HISTORY.clear()
    agent.NODES = 0
    agent.DEADLINE = math.inf
    agent._GAME_BOARD = board.copy(stack=False)
    agent._GAME_COUNTS = {agent._rep_key(board): 1}
    agent._PATH_COUNTS = {}
    agent._NULL_SEARCH_ACTIVE = 0


def fixed_depth_move(board, depth):
    reset_for_root(board)
    move, score = agent.search_root(board, depth, -agent.INF, agent.INF)
    if move is None:
        raise RuntimeError("search returned no move in nonterminal sanity position")
    return move.uci(), score, agent.NODES


def main():
    cases = [
        (
            "rook_takes_free_queen",
            make_board([
                ("g1", "K"), ("e1", "R"),
                ("g8", "k"), ("e5", "q"),
            ]),
            {"e1e5"},
        ),
        (
            "bishop_takes_free_queen",
            make_board([
                ("g1", "K"), ("c4", "B"),
                ("g8", "k"), ("f7", "q"),
            ]),
            {"c4f7"},
        ),
        (
            "knight_takes_free_queen",
            make_board([
                ("g1", "K"), ("f3", "N"),
                ("g8", "k"), ("e5", "q"),
            ]),
            {"f3e5"},
        ),
        (
            "queen_takes_free_rook",
            make_board([
                ("g1", "K"), ("d1", "Q"),
                ("g8", "k"), ("d5", "r"),
            ]),
            {"d1d5"},
        ),
        (
            "promotion_capture",
            make_board([
                ("e1", "K"), ("g7", "P"),
                ("e8", "k"), ("h8", "r"),
            ]),
            {"g7h8q", "g7h8r", "g7h8b", "g7h8n"},
        ),
        (
            "king_captures_checking_queen",
            make_board([
                ("e1", "K"),
                ("g8", "k"), ("e2", "q"),
            ]),
            {"e1e2"},
        ),
    ]

    failures = []
    print("CAPTURE SANITY SUITE")
    print("=" * 72)

    for name, board, expected in cases:
        for depth in (1, 3):
            move, score, nodes = fixed_depth_move(board.copy(stack=False), depth)
            ok = move in expected
            print(
                f"{name:34s} depth={depth} move={move:6s} "
                f"score={score:+6d} nodes={nodes:7d} "
                f"{'PASS' if ok else 'FAIL'}"
            )
            if not ok:
                failures.append((name, depth, move, sorted(expected), board.fen()))

    if failures:
        print("\nFAILURES")
        for row in failures:
            print(row)
        raise SystemExit(1)

    print("\nALL CAPTURE SANITY CHECKS PASSED")


if __name__ == "__main__":
    main()

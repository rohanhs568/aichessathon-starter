"""Production exact-threefold protocol/history test for the candidate agent."""
from __future__ import annotations

import chess
import agent


def reset():
    agent._GAME_BOARD = None
    agent._GAME_COUNTS = {}
    agent._PATH_COUNTS = {}
    agent._NULL_SEARCH_ACTIVE = 0
    agent.TT.clear()


def persist_our_move(root, uci):
    move = chess.Move.from_uci(uci)
    assert move in root.legal_moves
    after = root.copy(stack=False)
    after.push(move)
    agent._GAME_BOARD = after.copy(stack=False)
    agent._record_game_position(agent._GAME_BOARD)
    return after


def build_counts(board):
    counts = {}
    replay = board.root()
    counts[agent._rep_key(replay)] = 1
    for move in board.move_stack:
        replay.push(move)
        key = agent._rep_key(replay)
        counts[key] = counts.get(key, 0) + 1
    return counts


def main():
    reset()
    actual = chess.Board()
    root = agent._sync_game_board(actual.fen())
    start_key = agent._rep_key(actual)

    white_moves = ["g1f3", "f3g1", "g1f3", "f3g1"]
    black_moves = ["g8f6", "f6g8", "g8f6", "f6g8"]

    for pair, (ours, theirs) in enumerate(zip(white_moves, black_moves), 1):
        root = agent._sync_game_board(actual.fen())
        persist_our_move(root, ours)
        actual.push(chess.Move.from_uci(ours))

        actual.push(chess.Move.from_uci(theirs))
        synced = agent._sync_game_board(actual.fen())

        expected = build_counts(actual)
        if agent._GAME_COUNTS != expected:
            raise SystemExit(
                f"FAIL protocol counts after pair {pair}: "
                f"expected start={expected.get(start_key)}, "
                f"observed start={agent._GAME_COUNTS.get(start_key)}"
            )

        print(
            f"pair={pair} plies={len(actual.move_stack)} "
            f"start_occurrences={agent._GAME_COUNTS.get(start_key, 0)}"
        )

    if agent._GAME_COUNTS.get(start_key) != 3:
        raise SystemExit("FAIL did not reconstruct third occurrence")
    if not actual.can_claim_threefold_repetition():
        raise SystemExit("FAIL python-chess ground truth is not threefold")

    # Current root itself is now a third occurrence and must score as a draw.
    agent._PATH_COUNTS = {}
    if not agent._is_repetition_draw(synced):
        raise SystemExit("FAIL production repetition detector missed third occurrence")

    # One full knight cycle creates only a second occurrence, not a draw.
    reset()
    one = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]:
        one.push(chess.Move.from_uci(uci))
    agent._GAME_COUNTS = build_counts(one)
    agent._PATH_COUNTS = {}
    if agent._is_repetition_draw(one):
        raise SystemExit("FAIL second occurrence incorrectly treated as exact threefold")

    print("EXACT REPETITION PRODUCTION TEST PASSED")


if __name__ == "__main__":
    main()

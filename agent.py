"""Chessathon classical search agent."""

import time

import chess
import chess.polyglot


PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

MATE = 1_000_000
INF = float("inf")

# Transposition-table entry types.
EXACT = 0
LOWER = 1
UPPER = 2

# Persists between moves during one game.
TT = {}
MAX_TT_SIZE = 250_000

NODES = 0
DEADLINE = 0.0


class SearchTimeout(Exception):
    pass


def evaluate(board):
    """Material evaluation from the side-to-move perspective."""
    side = board.turn
    score = 0

    for piece, value in PIECE_VALUE.items():
        my_pieces = len(board.pieces(piece, side))
        their_pieces = len(board.pieces(piece, not side))

        score += value * (my_pieces - their_pieces)

    return score


def position_key(board):
    """
    Hash the board for the transposition table.

    Include halfmove_clock because positions with identical pieces
    can differ with respect to the fifty-move rule.
    """
    return (
        chess.polyglot.zobrist_hash(board),
        board.halfmove_clock,
    )


def move_order_score(board, move, tt_move):
    """
    Larger score = search this move earlier.

    Priority:
        1. previous TT best move
        2. promotions
        3. captures, especially valuable victim / cheap attacker
        4. quiet moves
    """

    if tt_move is not None and move == tt_move:
        return 10_000_000

    score = 0

    if move.promotion is not None:
        score += 1_000_000
        score += PIECE_VALUE.get(move.promotion, 0)

    if board.is_capture(move):
        if board.is_en_passant(move):
            victim_value = PIECE_VALUE[chess.PAWN]
        else:
            victim = board.piece_at(move.to_square)
            victim_value = (
                PIECE_VALUE.get(victim.piece_type, 0)
                if victim is not None
                else 0
            )

        attacker = board.piece_at(move.from_square)
        attacker_value = (
            PIECE_VALUE.get(attacker.piece_type, 0)
            if attacker is not None
            else 0
        )

        # MVV-LVA:
        # Most Valuable Victim, Least Valuable Attacker.
        score += 100_000
        score += 10 * victim_value - attacker_value

    return score


def ordered_moves(board, tt_move=None):
    moves = list(board.legal_moves)

    moves.sort(
        key=lambda move: move_order_score(board, move, tt_move),
        reverse=True,
    )

    return moves


def check_time():
    """
    Avoid calling the clock at every node because that itself costs time.
    """
    if NODES % 1024 == 0 and time.monotonic() >= DEADLINE:
        raise SearchTimeout


def negamax(board, depth, alpha, beta, ply):
    global NODES

    NODES += 1
    check_time()

    key = position_key(board)

    # -----------------------------
    # TRANSPOSITION TABLE LOOKUP
    # -----------------------------

    entry = TT.get(key)
    tt_move = None

    if entry is not None:
        entry_depth, entry_score, entry_flag, entry_move = entry
        tt_move = entry_move

        if entry_depth >= depth:
            if entry_flag == EXACT:
                return entry_score

            if entry_flag == LOWER:
                alpha = max(alpha, entry_score)

            elif entry_flag == UPPER:
                beta = min(beta, entry_score)

            if alpha >= beta:
                return entry_score

    # These are the bounds for the search we are actually about to perform.
    alpha_start = alpha
    beta_start = beta

    moves = ordered_moves(board, tt_move)

    # Terminal position.
    if not moves:
        if board.is_check():
            # Prefer faster mates and postpone being mated.
            return -MATE + ply

        return 0

    # Search horizon.
    if depth == 0:
        return evaluate(board)

    best_score = -INF
    best_move = None

    for move in moves:
        board.push(move)

        try:
            score = -negamax(
                board,
                depth - 1,
                -beta,
                -alpha,
                ply + 1,
            )
        finally:
            # Important: restore the board even if the clock interrupts search.
            board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, score)

        if alpha >= beta:
            break

    # -----------------------------
    # TRANSPOSITION TABLE STORE
    # -----------------------------

    if best_score <= alpha_start:
        flag = UPPER
    elif best_score >= beta_start:
        flag = LOWER
    else:
        flag = EXACT

    TT[key] = (
        depth,
        best_score,
        flag,
        best_move,
    )

    return best_score


def search_root(board, depth):
    alpha = -INF
    beta = INF

    best_score = -INF
    best_move = None

    entry = TT.get(position_key(board))
    tt_move = entry[3] if entry is not None else None

    moves = ordered_moves(board, tt_move)

    for move in moves:
        # Root checks the clock every move as well.
        if time.monotonic() >= DEADLINE:
            raise SearchTimeout

        board.push(move)

        try:
            score = -negamax(
                board,
                depth - 1,
                -beta,
                -alpha,
                1,
            )
        finally:
            board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, score)

    return best_move, best_score


def choose_time_budget(time_left_ms):
    """
    Conservative first-pass clock management.

    Spend more when we have lots of time, but preserve a reserve.
    """

    if time_left_ms <= 500:
        return max(10.0, 0.20 * time_left_ms)

    # Roughly use 3% of remaining clock plus most of the 0.5 s increment.
    budget_ms = 0.03 * time_left_ms + 350

    # Don't spend huge amounts on one ordinary move yet.
    budget_ms = min(budget_ms, 3000)

    # Keep at least ~500 ms in reserve whenever possible.
    budget_ms = min(
        budget_ms,
        max(50, time_left_ms - 500),
    )

    return budget_ms


def get_move(fen: str, time_left_ms: int) -> str:
    global NODES
    global DEADLINE
    global TT

    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)

    # Normally the platform should not ask us to move in a terminal position.
    if not legal_moves:
        raise ValueError("get_move called on terminal position")

    # Prevent an unbounded Python dictionary from eventually consuming too much RAM.
    if len(TT) > MAX_TT_SIZE:
        TT.clear()

    budget_ms = choose_time_budget(time_left_ms)

    DEADLINE = time.monotonic() + budget_ms / 1000.0
    NODES = 0

    # Emergency fallback. We can always return this if the first search times out.
    best_move = legal_moves[0]
    best_score = None

    depth = 1

    while True:
        start_nodes = NODES
        start_time = time.monotonic()

        try:
            move, score = search_root(board, depth)

        except SearchTimeout:
            break

        # Only accept the result from a COMPLETED depth.
        best_move = move
        best_score = score

        elapsed = time.monotonic() - start_time
        depth_nodes = NODES - start_nodes

        print(
            f"depth={depth} "
            f"score={best_score} "
            f"nodes={depth_nodes} "
            f"total_nodes={NODES} "
            f"time={elapsed:.3f}s"
        )

        depth += 1

    print(
        f"playing={best_move.uci()} "
        f"completed_depth={depth - 1} "
        f"total_nodes={NODES}"
    )

    return best_move.uci()
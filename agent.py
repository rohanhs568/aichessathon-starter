"""Chessathon classical search agent."""

import time

import chess
import chess.polyglot


# ============================================================
# CONSTANTS
# ============================================================

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

MATE = 1_000_000
INF = float("inf")

EXACT = 0
LOWER = 1
UPPER = 2

MAX_TT_SIZE = 250_000


# ============================================================
# GLOBAL SEARCH STATE
# ============================================================

TT = {}

# Two killer moves for each ply.
KILLERS = {}

# (piece_type, from_square, to_square) -> score
HISTORY = {}

NODES = 0
DEADLINE = 0.0


class SearchTimeout(Exception):
    pass


# ============================================================
# EVALUATION
# ============================================================

def evaluate(board):
    """
    Material-only evaluation from the perspective
    of the side to move.
    """

    side = board.turn
    score = 0

    for piece, value in PIECE_VALUE.items():
        my_pieces = len(board.pieces(piece, side))
        their_pieces = len(board.pieces(piece, not side))

        score += value * (my_pieces - their_pieces)

    return score


# ============================================================
# TRANSPOSITION TABLE
# ============================================================

def position_key(board):
    return (
        chess.polyglot.zobrist_hash(board),
        board.halfmove_clock,
    )


# ============================================================
# TIME MANAGEMENT
# ============================================================

def check_time():
    """
    Checking the system clock itself has a cost,
    so only check periodically.
    """

    if NODES % 1024 == 0:
        if time.monotonic() >= DEADLINE:
            raise SearchTimeout


def choose_time_budget(time_left_ms):
    """
    Simple conservative clock management.
    """

    if time_left_ms <= 500:
        return max(10.0, 0.20 * time_left_ms)

    budget_ms = 0.03 * time_left_ms + 350

    budget_ms = min(budget_ms, 3000)

    budget_ms = min(
        budget_ms,
        max(50, time_left_ms - 500),
    )

    return budget_ms


# ============================================================
# MOVE ORDERING
# ============================================================

def move_order_score(board, move, tt_move, ply):
    """
    Larger score = search earlier.

    Priority roughly:

        TT move
        promotions
        captures
        killer moves
        history heuristic
        other quiet moves
    """

    # Best move remembered from an earlier search.
    if tt_move is not None and move == tt_move:
        return 100_000_000

    score = 0

    # --------------------------------------------------------
    # Promotions
    # --------------------------------------------------------

    if move.promotion is not None:
        score += 10_000_000
        score += PIECE_VALUE.get(move.promotion, 0)

    # --------------------------------------------------------
    # Captures: MVV-LVA
    # --------------------------------------------------------

    if board.is_capture(move):

        if board.is_en_passant(move):
            victim_value = PIECE_VALUE[chess.PAWN]

        else:
            victim = board.piece_at(move.to_square)

            if victim is None:
                victim_value = 0
            else:
                victim_value = PIECE_VALUE.get(
                    victim.piece_type,
                    0,
                )

        attacker = board.piece_at(move.from_square)

        if attacker is None:
            attacker_value = 0
        else:
            attacker_value = PIECE_VALUE.get(
                attacker.piece_type,
                0,
            )

        score += 1_000_000
        score += 10 * victim_value - attacker_value

        return score

    # --------------------------------------------------------
    # Killer moves
    # --------------------------------------------------------

    killers = KILLERS.get(ply, [])

    if len(killers) > 0 and move == killers[0]:
        score += 900_000

    elif len(killers) > 1 and move == killers[1]:
        score += 800_000

    # --------------------------------------------------------
    # History heuristic
    # --------------------------------------------------------

    piece = board.piece_at(move.from_square)

    if piece is not None:
        key = (
            piece.piece_type,
            move.from_square,
            move.to_square,
        )

        score += HISTORY.get(key, 0)

    return score


def ordered_moves(board, tt_move=None, ply=0):
    moves = list(board.legal_moves)

    moves.sort(
        key=lambda move: move_order_score(
            board,
            move,
            tt_move,
            ply,
        ),
        reverse=True,
    )

    return moves


# ============================================================
# KILLER + HISTORY UPDATES
# ============================================================

def record_quiet_cutoff(board, move, depth, ply):
    """
    A quiet move produced a beta cutoff.

    Remember it as a useful move-ordering candidate.
    """

    if board.is_capture(move):
        return

    if move.promotion is not None:
        return

    # --------------------------------------------------------
    # Killer heuristic
    # --------------------------------------------------------

    killers = KILLERS.setdefault(ply, [])

    if move not in killers:
        killers.insert(0, move)

        if len(killers) > 2:
            killers.pop()

    # --------------------------------------------------------
    # History heuristic
    # --------------------------------------------------------

    piece = board.piece_at(move.from_square)

    if piece is not None:
        key = (
            piece.piece_type,
            move.from_square,
            move.to_square,
        )

        HISTORY[key] = HISTORY.get(key, 0) + depth * depth


# ============================================================
# QUIESCENCE SEARCH
# ============================================================

def quiescence(board, alpha, beta, ply):
    global NODES

    NODES += 1
    check_time()

    moves = list(board.legal_moves)

    # --------------------------------------------------------
    # Terminal positions
    # --------------------------------------------------------

    if not moves:

        if board.is_check():
            return -MATE + ply

        return 0

    # --------------------------------------------------------
    # If in check, standing pat is illegal.
    # Search every legal evasion.
    # --------------------------------------------------------

    if board.is_check():

        best_score = -INF

        for move in ordered_moves(board, ply=ply):

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

            best_score = max(best_score, score)
            alpha = max(alpha, score)

            if alpha >= beta:
                break

        return best_score

    # --------------------------------------------------------
    # Stand pat
    # --------------------------------------------------------

    stand_pat = evaluate(board)

    if stand_pat >= beta:
        return stand_pat

    alpha = max(alpha, stand_pat)

    # --------------------------------------------------------
    # Search tactical continuations only
    # --------------------------------------------------------

    tactical_moves = [
        move
        for move in moves
        if board.is_capture(move)
        or move.promotion is not None
    ]

    tactical_moves.sort(
        key=lambda move: move_order_score(
            board,
            move,
            None,
            ply,
        ),
        reverse=True,
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

        alpha = max(alpha, score)

    return alpha


# ============================================================
# NEGAMAX + ALPHA-BETA + PVS
# ============================================================

def negamax(board, depth, alpha, beta, ply):
    global NODES

    NODES += 1
    check_time()

    key = position_key(board)

    # --------------------------------------------------------
    # TRANSPOSITION TABLE LOOKUP
    # --------------------------------------------------------

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

    alpha_start = alpha
    beta_start = beta

    moves = ordered_moves(
        board,
        tt_move,
        ply,
    )

    # --------------------------------------------------------
    # TERMINAL POSITION
    # --------------------------------------------------------

    if not moves:

        if board.is_check():
            return -MATE + ply

        return 0

    # --------------------------------------------------------
    # SEARCH HORIZON
    # --------------------------------------------------------

    if depth == 0:
        return quiescence(
            board,
            alpha,
            beta,
            ply,
        )

    best_score = -INF
    best_move = None

    first_move = True

    # --------------------------------------------------------
    # SEARCH MOVES
    # --------------------------------------------------------

    for move in moves:

        board.push(move)

        try:

            if first_move:

                # Principal variation candidate:
                # search normally.
                score = -negamax(
                    board,
                    depth - 1,
                    -beta,
                    -alpha,
                    ply + 1,
                )

                first_move = False

            else:

                # --------------------------------------------
                # PVS
                #
                # First ask the cheaper question:
                #
                # "Can this move beat alpha at all?"
                #
                # Search with a tiny window.
                # --------------------------------------------

                score = -negamax(
                    board,
                    depth - 1,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                )

                # If it does beat alpha, we need its
                # actual value, so re-search fully.
                if alpha < score < beta:

                    score = -negamax(
                        board,
                        depth - 1,
                        -beta,
                        -alpha,
                        ply + 1,
                    )

        finally:
            board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, score)

        if alpha >= beta:

            record_quiet_cutoff(
                board,
                move,
                depth,
                ply,
            )

            break

    # --------------------------------------------------------
    # TRANSPOSITION TABLE STORE
    # --------------------------------------------------------

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


# ============================================================
# ROOT SEARCH
# ============================================================

def search_root(board, depth):
    alpha = -INF
    beta = INF

    best_score = -INF
    best_move = None

    entry = TT.get(position_key(board))

    if entry is not None:
        tt_move = entry[3]
    else:
        tt_move = None

    moves = ordered_moves(
        board,
        tt_move,
        0,
    )

    first_move = True

    for move in moves:

        if time.monotonic() >= DEADLINE:
            raise SearchTimeout

        board.push(move)

        try:

            if first_move:

                score = -negamax(
                    board,
                    depth - 1,
                    -beta,
                    -alpha,
                    1,
                )

                first_move = False

            else:

                # PVS narrow search.
                score = -negamax(
                    board,
                    depth - 1,
                    -alpha - 1,
                    -alpha,
                    1,
                )

                # Re-search if it genuinely improves alpha.
                if alpha < score < beta:

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


# ============================================================
# ENTRYPOINT
# ============================================================

def get_move(fen: str, time_left_ms: int) -> str:
    global NODES
    global DEADLINE
    global TT

    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        raise ValueError(
            "get_move called on terminal position"
        )

    # Prevent unlimited TT growth.
    if len(TT) > MAX_TT_SIZE:
        TT.clear()

    budget_ms = choose_time_budget(time_left_ms)

    DEADLINE = (
        time.monotonic()
        + budget_ms / 1000.0
    )

    NODES = 0

    # Emergency fallback.
    best_move = legal_moves[0]
    best_score = None

    depth = 1

    # --------------------------------------------------------
    # ITERATIVE DEEPENING
    # --------------------------------------------------------

    while True:

        start_nodes = NODES
        start_time = time.monotonic()

        try:

            move, score = search_root(
                board,
                depth,
            )

        except SearchTimeout:
            break

        # Only keep results from completed iterations.
        best_move = move
        best_score = score

        elapsed = (
            time.monotonic()
            - start_time
        )

        depth_nodes = (
            NODES
            - start_nodes
        )

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
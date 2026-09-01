"""The submission entrypoint. The platform imports this file and calls get_move."""

import chess

# Import time runs once per game, inside a 60 second budget, before your clock starts.
# Load weights and build tables out here, not inside get_move.

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

MATE = 1_000_000
SEARCH_DEPTH = 2


def evaluate(board):
    side = board.turn
    score = 0

    for piece, value in PIECE_VALUE.items():
        my_pieces = len(board.pieces(piece, side))
        their_pieces = len(board.pieces(piece, not side))

        score += value * (my_pieces - their_pieces)

    return score

def negamax(board, depth, alpha, beta):
    moves = list(board.legal_moves)

    # No legal moves means checkmate or stalemate.
    if not moves:
        if board.is_check():
            return -MATE
        return 0

    # We have reached the search horizon.
    if depth == 0:
        return evaluate(board)

    best_score = -float("inf")

    for move in moves:
        board.push(move)

        score = -negamax(
        board,
        depth - 1,
        -beta,
        -alpha,
    )

        board.pop()

        best_score = max(best_score, score)
        alpha = max(alpha, score)

        if alpha >= beta:
            break

    return best_score


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)

    best_move = None
    best_score = -float("inf")

    alpha = -float("inf")
    beta = float("inf")

    for move in board.legal_moves:
        board.push(move)

        score = -negamax(
            board,
            SEARCH_DEPTH - 1,
            -beta,
            -alpha,
        )

        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, score)

    return best_move.uci()
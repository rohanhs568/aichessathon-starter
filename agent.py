"""Chessathon learned-evaluation search agent.

Search:
    iterative deepening
    alpha-beta negamax
    principal variation search (PVS)
    transposition table with mate-distance normalization
    quiescence search
    TT / MVV-LVA / killer / history move ordering
    null-move pruning
    late-move reductions (LMR)
    conservative frontier futility pruning
    aspiration windows

Evaluation:
    V1 dual-perspective NNUE-style network exported to weights/v1.npz
    772 sparse features (piece-square + castling rights)
    horizontal king mirroring
    shared 64-wide feature transformer
    SCReLU
    8 piece-count output buckets

The network was trained with target tanh(cp / K). For search we do not need to
compute tanh followed by atanh: the pre-tanh output is directly cp / K, so the
runtime score is simply raw_output * K.
"""

from __future__ import annotations

import os

# The tournament gives one CPU core. Prevent tiny BLAS operations from trying
# to create extra worker threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import math
import time
from pathlib import Path

import chess
import chess.polyglot
import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - tournament image includes numba
    njit = None


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
MATE_THRESHOLD = 900_000
INF = 2_000_000

EXACT = 0
LOWER = 1
UPPER = 2

MAX_TT_SIZE = 300_000
MAX_EVAL_CACHE_SIZE = 120_000
TIME_CHECK_MASK = 255  # Check clock every 256 visited nodes.

# V1 feature layout. These must match training/train_v1.py.
PIECE_SQUARE_FEATURES = 2 * 6 * 64
CASTLING_BASE = PIECE_SQUARE_FEATURES
INPUT_FEATURES = PIECE_SQUARE_FEATURES + 4
MAX_ACTIVE_FEATURES = 36

# Conservative pruning settings. These are intentionally simple initial
# values, suitable for testing before any Elo tuning.
FUTILITY_MARGIN_D1 = 160


# ============================================================
# MODEL LOADING
# ============================================================

MODEL_PATH = Path(__file__).resolve().parent / "weights" / "v1.npz"

NN_AVAILABLE = False
NN_ERROR = None
EMBEDDING = None
HIDDEN_BIAS = None
OUTPUT_WEIGHT = None
OUTPUT_BIAS = None
HIDDEN = 0
BUCKETS = 0
K_CP = 400.0


def _load_weights() -> None:
    global NN_AVAILABLE
    global NN_ERROR
    global EMBEDDING
    global HIDDEN_BIAS
    global OUTPUT_WEIGHT
    global OUTPUT_BIAS
    global HIDDEN
    global BUCKETS
    global K_CP

    try:
        with np.load(MODEL_PATH, allow_pickle=False) as data:
            embedding = np.asarray(data["embedding"], dtype=np.float32).copy()
            hidden_bias = np.asarray(data["hidden_bias"], dtype=np.float32).copy()
            output_weight = np.asarray(data["output_weight"], dtype=np.float32).copy()
            output_bias = np.asarray(data["output_bias"], dtype=np.float32).copy()

            input_features = int(np.asarray(data["input_features"]).item())
            hidden = int(np.asarray(data["hidden"]).item())
            buckets = int(np.asarray(data["buckets"]).item())
            k_cp = float(np.asarray(data["k_cp"]).item())

        if input_features != INPUT_FEATURES:
            raise RuntimeError(
                f"weight input_features={input_features}, expected {INPUT_FEATURES}"
            )
        if embedding.shape != (INPUT_FEATURES, hidden):
            raise RuntimeError(f"bad embedding shape {embedding.shape}")
        if hidden_bias.shape != (hidden,):
            raise RuntimeError(f"bad hidden_bias shape {hidden_bias.shape}")
        if output_weight.shape != (buckets, 2 * hidden):
            raise RuntimeError(f"bad output_weight shape {output_weight.shape}")
        if output_bias.shape != (buckets,):
            raise RuntimeError(f"bad output_bias shape {output_bias.shape}")
        if buckets < 1:
            raise RuntimeError("model has no output buckets")

        EMBEDDING = np.ascontiguousarray(embedding)
        HIDDEN_BIAS = np.ascontiguousarray(hidden_bias)
        OUTPUT_WEIGHT = np.ascontiguousarray(output_weight)
        OUTPUT_BIAS = np.ascontiguousarray(output_bias)
        HIDDEN = hidden
        BUCKETS = buckets
        K_CP = k_cp
        NN_AVAILABLE = True

        print(
            f"loaded learned evaluator: hidden={HIDDEN} "
            f"buckets={BUCKETS} k_cp={K_CP:g} "
            f"weights={MODEL_PATH.name}"
        )

    except Exception as exc:
        NN_AVAILABLE = False
        NN_ERROR = repr(exc)
        print(
            f"warning: learned evaluator unavailable ({NN_ERROR}); "
            "using material fallback"
        )


_load_weights()


# ============================================================
# NUMBA INFERENCE KERNEL
# ============================================================

if njit is not None:

    @njit(cache=False, fastmath=True)
    def _nn_kernel(
        stm_ids,
        stm_count,
        opp_ids,
        opp_count,
        piece_count,
        embedding,
        hidden_bias,
        output_weight,
        output_bias,
        buckets,
        k_cp,
    ):
        hidden = hidden_bias.shape[0]

        stm_hidden = np.empty(hidden, dtype=np.float32)
        opp_hidden = np.empty(hidden, dtype=np.float32)

        for j in range(hidden):
            s = hidden_bias[j]
            for i in range(stm_count):
                s += embedding[stm_ids[i], j]
            if s < 0.0:
                s = 0.0
            elif s > 1.0:
                s = 1.0
            stm_hidden[j] = s * s

            s = hidden_bias[j]
            for i in range(opp_count):
                s += embedding[opp_ids[i], j]
            if s < 0.0:
                s = 0.0
            elif s > 1.0:
                s = 1.0
            opp_hidden[j] = s * s

        if buckets == 1:
            bucket = 0
        else:
            bucket = (piece_count - 2) // 4
            if bucket < 0:
                bucket = 0
            elif bucket >= buckets:
                bucket = buckets - 1

        raw = output_bias[bucket]

        for j in range(hidden):
            raw += output_weight[bucket, j] * stm_hidden[j]
            raw += output_weight[bucket, hidden + j] * opp_hidden[j]

        # Training output = tanh(raw), target = tanh(cp / K).
        # Therefore inverse-target search score is simply raw * K.
        cp = raw * k_cp

        if cp > 4000.0:
            cp = 4000.0
        elif cp < -4000.0:
            cp = -4000.0

        return int(round(cp))

else:
    _nn_kernel = None


_STM_IDS = np.empty(MAX_ACTIVE_FEATURES, dtype=np.int16)
_OPP_IDS = np.empty(MAX_ACTIVE_FEATURES, dtype=np.int16)


def _numpy_nn_kernel(stm_count, opp_count, piece_count):
    """Fallback inference if Numba is unavailable."""
    stm = EMBEDDING[_STM_IDS[:stm_count]].sum(axis=0) + HIDDEN_BIAS
    opp = EMBEDDING[_OPP_IDS[:opp_count]].sum(axis=0) + HIDDEN_BIAS

    stm = np.clip(stm, 0.0, 1.0)
    opp = np.clip(opp, 0.0, 1.0)
    stm *= stm
    opp *= opp

    if BUCKETS == 1:
        bucket = 0
    else:
        bucket = (piece_count - 2) // 4
        bucket = min(max(bucket, 0), BUCKETS - 1)

    raw = float(OUTPUT_BIAS[bucket])
    raw += float(np.dot(OUTPUT_WEIGHT[bucket, :HIDDEN], stm))
    raw += float(np.dot(OUTPUT_WEIGHT[bucket, HIDDEN:], opp))

    return int(round(max(-4000.0, min(4000.0, raw * K_CP))))


if NN_AVAILABLE and _nn_kernel is not None:
    try:
        # Pay JIT compilation during import, inside the free 60 second init
        # budget, instead of on the first timed move.
        _dummy_ids = np.zeros(MAX_ACTIVE_FEATURES, dtype=np.int16)
        _nn_kernel(
            _dummy_ids,
            1,
            _dummy_ids,
            1,
            32,
            EMBEDDING,
            HIDDEN_BIAS,
            OUTPUT_WEIGHT,
            OUTPUT_BIAS,
            BUCKETS,
            K_CP,
        )
        print("numba evaluator ready")
    except Exception as exc:
        print(f"warning: numba warmup failed ({exc!r}); using NumPy inference")
        _nn_kernel = None


# ============================================================
# GLOBAL SEARCH STATE
# ============================================================

# key -> (depth, tt_score, flag, best_move)
TT = {}

# Static NN scores use only the chess position, not the halfmove clock.
EVAL_CACHE = {}

# Two killer moves for each search ply.
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


def material_evaluate(board):
    """Fast fallback score from the perspective of the side to move."""
    side = board.turn
    score = 0

    for piece_type, value in PIECE_VALUE.items():
        score += value * (
            len(board.pieces(piece_type, side))
            - len(board.pieces(piece_type, not side))
        )

    return score


def _append_castling_features(board, perspective, mirror_files, buffer, count):
    own_k = board.has_kingside_castling_rights(perspective)
    own_q = board.has_queenside_castling_rights(perspective)
    opp_k = board.has_kingside_castling_rights(not perspective)
    opp_q = board.has_queenside_castling_rights(not perspective)

    if mirror_files:
        own_k, own_q = own_q, own_k
        opp_k, opp_q = opp_q, opp_k

    if own_k:
        buffer[count] = CASTLING_BASE
        count += 1
    if own_q:
        buffer[count] = CASTLING_BASE + 1
        count += 1
    if opp_k:
        buffer[count] = CASTLING_BASE + 2
        count += 1
    if opp_q:
        buffer[count] = CASTLING_BASE + 3
        count += 1

    return count


def _encode_board_for_nn(board):
    """Fill the two sparse feature buffers exactly as training/train_v1.py."""
    stm = board.turn
    opp = not stm

    stm_king = board.king(stm)
    opp_king = board.king(opp)

    if stm_king is None or opp_king is None:
        return None

    stm_is_white = stm == chess.WHITE
    opp_is_white = opp == chess.WHITE

    mirror_stm = chess.square_file(stm_king) >= 4
    mirror_opp = chess.square_file(opp_king) >= 4

    stm_count = 0
    opp_count = 0
    piece_count = 0

    for square, piece in board.piece_map().items():
        piece_count += 1
        piece_index = piece.piece_type - 1
        rank = chess.square_rank(square)
        file_index = chess.square_file(square)

        # Side-to-move perspective.
        relative_colour = 0 if piece.color == stm else 1
        canonical_rank = rank if stm_is_white else 7 - rank
        canonical_file = 7 - file_index if mirror_stm else file_index
        canonical_square = canonical_rank * 8 + canonical_file
        _STM_IDS[stm_count] = (
            relative_colour * 6 * 64
            + piece_index * 64
            + canonical_square
        )
        stm_count += 1

        # Opponent perspective.
        relative_colour = 0 if piece.color == opp else 1
        canonical_rank = rank if opp_is_white else 7 - rank
        canonical_file = 7 - file_index if mirror_opp else file_index
        canonical_square = canonical_rank * 8 + canonical_file
        _OPP_IDS[opp_count] = (
            relative_colour * 6 * 64
            + piece_index * 64
            + canonical_square
        )
        opp_count += 1

    stm_count = _append_castling_features(
        board,
        stm,
        mirror_stm,
        _STM_IDS,
        stm_count,
    )
    opp_count = _append_castling_features(
        board,
        opp,
        mirror_opp,
        _OPP_IDS,
        opp_count,
    )

    if stm_count > MAX_ACTIVE_FEATURES or opp_count > MAX_ACTIVE_FEATURES:
        return None

    return stm_count, opp_count, piece_count


def evaluate(board):
    """Learned evaluation from the perspective of the side to move."""
    if not NN_AVAILABLE:
        return material_evaluate(board)

    zobrist = chess.polyglot.zobrist_hash(board)
    cached = EVAL_CACHE.get(zobrist)
    if cached is not None:
        return cached

    encoded = _encode_board_for_nn(board)
    if encoded is None:
        score = material_evaluate(board)
    else:
        stm_count, opp_count, piece_count = encoded

        if _nn_kernel is not None:
            score = _nn_kernel(
                _STM_IDS,
                stm_count,
                _OPP_IDS,
                opp_count,
                piece_count,
                EMBEDDING,
                HIDDEN_BIAS,
                OUTPUT_WEIGHT,
                OUTPUT_BIAS,
                BUCKETS,
                K_CP,
            )
        else:
            score = _numpy_nn_kernel(stm_count, opp_count, piece_count)

    if len(EVAL_CACHE) >= MAX_EVAL_CACHE_SIZE:
        EVAL_CACHE.clear()
    EVAL_CACHE[zobrist] = score

    return score


# ============================================================
# TRANSPOSITION TABLE
# ============================================================


def position_key(board):
    return (
        chess.polyglot.zobrist_hash(board),
        board.halfmove_clock,
    )


def score_to_tt(score, ply):
    """Make mate scores independent of the path length used to reach a node."""
    if score >= MATE_THRESHOLD:
        return score + ply
    if score <= -MATE_THRESHOLD:
        return score - ply
    return score


def score_from_tt(score, ply):
    if score >= MATE_THRESHOLD:
        return score - ply
    if score <= -MATE_THRESHOLD:
        return score + ply
    return score


# ============================================================
# TIME MANAGEMENT
# ============================================================


def check_time(force=False):
    if force or (NODES & TIME_CHECK_MASK) == 0:
        if time.monotonic() >= DEADLINE:
            raise SearchTimeout


def choose_time_budget(time_left_ms):
    """Conservative budget for the 120s + 0.5s tournament clock."""
    if time_left_ms <= 250:
        return max(5.0, 0.10 * time_left_ms)

    if time_left_ms <= 1_000:
        return max(15.0, 0.12 * time_left_ms)

    if time_left_ms <= 5_000:
        return min(350.0, 0.055 * time_left_ms + 60.0)

    # Around 3.4s from the initial 120s clock, then naturally less later.
    budget_ms = 0.025 * time_left_ms + 400.0
    budget_ms = min(budget_ms, 3_500.0)

    # Always leave a reserve so a timeout check cannot flag the process.
    budget_ms = min(budget_ms, max(25.0, time_left_ms - 250.0))
    return budget_ms


# ============================================================
# MOVE ORDERING
# ============================================================


def capture_values(board, move):
    if board.is_en_passant(move):
        victim_value = PIECE_VALUE[chess.PAWN]
    else:
        victim = board.piece_at(move.to_square)
        victim_value = 0 if victim is None else PIECE_VALUE.get(victim.piece_type, 0)

    attacker = board.piece_at(move.from_square)
    attacker_value = 0 if attacker is None else PIECE_VALUE.get(attacker.piece_type, 0)

    return victim_value, attacker_value


def move_order_score(board, move, tt_move, ply):
    if tt_move is not None and move == tt_move:
        return 100_000_000

    score = 0

    if move.promotion is not None:
        score += 10_000_000 + PIECE_VALUE.get(move.promotion, 0)

    if board.is_capture(move):
        victim_value, attacker_value = capture_values(board, move)
        return score + 1_000_000 + 10 * victim_value - attacker_value

    killers = KILLERS.get(ply)
    if killers:
        if move == killers[0]:
            score += 900_000
        elif len(killers) > 1 and move == killers[1]:
            score += 800_000

    piece = board.piece_at(move.from_square)
    if piece is not None:
        key = (piece.piece_type, move.from_square, move.to_square)
        score += HISTORY.get(key, 0)

    return score


def order_move_list(board, moves, tt_move=None, ply=0):
    moves.sort(
        key=lambda move: move_order_score(board, move, tt_move, ply),
        reverse=True,
    )
    return moves


def ordered_moves(board, tt_move=None, ply=0):
    return order_move_list(board, list(board.legal_moves), tt_move, ply)


# ============================================================
# KILLER + HISTORY UPDATES
# ============================================================


def record_quiet_cutoff(board, move, depth, ply):
    if board.is_capture(move) or move.promotion is not None:
        return

    killers = KILLERS.setdefault(ply, [])
    if move not in killers:
        killers.insert(0, move)
        if len(killers) > 2:
            killers.pop()

    piece = board.piece_at(move.from_square)
    if piece is not None:
        key = (piece.piece_type, move.from_square, move.to_square)
        HISTORY[key] = min(
            1_000_000,
            HISTORY.get(key, 0) + depth * depth,
        )


def decay_history():
    if not HISTORY:
        return
    for key in list(HISTORY):
        value = HISTORY[key] // 2
        if value:
            HISTORY[key] = value
        else:
            del HISTORY[key]


# ============================================================
# SEARCH HELPERS
# ============================================================


def has_non_pawn_material(board, colour):
    return bool(
        board.pieces(chess.KNIGHT, colour)
        or board.pieces(chess.BISHOP, colour)
        or board.pieces(chess.ROOK, colour)
        or board.pieces(chess.QUEEN, colour)
    )


def lmr_reduction(depth, move_index):
    """Small, conservative late-move reduction in plies."""
    reduction = 1
    if depth >= 6 and move_index >= 6:
        reduction = 2
    if depth >= 9 and move_index >= 12:
        reduction = 3
    return min(reduction, depth - 2)


# ============================================================
# QUIESCENCE SEARCH
# ============================================================


def quiescence(board, alpha, beta, ply):
    global NODES

    NODES += 1
    check_time()

    moves = list(board.legal_moves)

    if not moves:
        if board.is_check():
            return -MATE + ply
        return 0

    if board.halfmove_clock >= 100:
        return 0

    if board.is_check():
        best_score = -INF
        order_move_list(board, moves, None, ply)

        for move in moves:
            board.push(move)
            try:
                score = -quiescence(board, -beta, -alpha, ply + 1)
            finally:
                board.pop()

            if score > best_score:
                best_score = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        return best_score

    stand_pat = evaluate(board)

    if stand_pat >= beta:
        return stand_pat
    if stand_pat > alpha:
        alpha = stand_pat

    tactical_moves = [
        move
        for move in moves
        if board.is_capture(move) or move.promotion is not None
    ]
    order_move_list(board, tactical_moves, None, ply)

    for move in tactical_moves:
        board.push(move)
        try:
            score = -quiescence(board, -beta, -alpha, ply + 1)
        finally:
            board.pop()

        if score >= beta:
            return score
        if score > alpha:
            alpha = score

    return alpha


# ============================================================
# NEGAMAX + ALPHA-BETA + PVS + PRUNING
# ============================================================


def negamax(board, depth, alpha, beta, ply, allow_null=True):
    global NODES

    NODES += 1
    check_time()

    if depth <= 0:
        return quiescence(board, alpha, beta, ply)

    key = position_key(board)
    entry = TT.get(key)
    tt_move = None

    if entry is not None:
        entry_depth, stored_score, entry_flag, entry_move = entry
        entry_score = score_from_tt(stored_score, ply)
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
    in_check = board.is_check()

    moves = list(board.legal_moves)

    if not moves:
        if in_check:
            return -MATE + ply
        return 0

    if board.halfmove_clock >= 100:
        return 0

    # --------------------------------------------------------
    # Null-move pruning
    # --------------------------------------------------------
    # If we can pass and still exceed beta, a normal move is very likely to
    # exceed beta as well. Disable consecutive null moves and avoid likely
    # pawn-only zugzwang positions.
    null_static_eval = None
    if (
        allow_null
        and depth >= 3
        and not in_check
        and beta < MATE_THRESHOLD
        and has_non_pawn_material(board, board.turn)
    ):
        # Standard safety gate: only try null move when the static position
        # already looks at least as good as beta. This is more conservative
        # than unconditional null pruning and reduces zugzwang/tactical risk.
        null_static_eval = evaluate(board)

    if null_static_eval is not None and null_static_eval >= beta:
        reduction = 2 + depth // 6
        null_depth = max(0, depth - 1 - reduction)

        board.push(chess.Move.null())
        try:
            null_score = -negamax(
                board,
                null_depth,
                -beta,
                -beta + 1,
                ply + 1,
                allow_null=False,
            )
        finally:
            board.pop()

        if null_score >= beta and null_score < MATE_THRESHOLD:
            return null_score

    order_move_list(board, moves, tt_move, ply)

    # Frontier futility only. Our learned evaluator is still noisy enough that
    # aggressive multi-ply futility pruning would be premature.
    static_eval = None
    use_futility = depth == 1 and not in_check and alpha > -MATE_THRESHOLD
    if use_futility:
        static_eval = evaluate(board)

    best_score = -INF
    best_move = None
    searched_moves = 0

    for move_index, move in enumerate(moves):
        is_capture = board.is_capture(move)
        is_promotion = move.promotion is not None
        quiet = not is_capture and not is_promotion
        gives_check = board.gives_check(move)

        if (
            use_futility
            and searched_moves > 0
            and quiet
            and not gives_check
            and static_eval + FUTILITY_MARGIN_D1 <= alpha
        ):
            continue

        board.push(move)
        try:
            if searched_moves == 0:
                score = -negamax(
                    board,
                    depth - 1,
                    -beta,
                    -alpha,
                    ply + 1,
                    allow_null=True,
                )
            else:
                can_reduce = (
                    depth >= 3
                    and move_index >= 3
                    and quiet
                    and not in_check
                    and not gives_check
                )

                if can_reduce:
                    reduction = lmr_reduction(depth, move_index)
                    reduced_depth = max(0, depth - 1 - reduction)
                    score = -negamax(
                        board,
                        reduced_depth,
                        -alpha - 1,
                        -alpha,
                        ply + 1,
                        allow_null=True,
                    )

                    # A reduced move that looks interesting earns its full
                    # depth back before it can change the principal variation.
                    if score > alpha:
                        score = -negamax(
                            board,
                            depth - 1,
                            -alpha - 1,
                            -alpha,
                            ply + 1,
                            allow_null=True,
                        )
                else:
                    score = -negamax(
                        board,
                        depth - 1,
                        -alpha - 1,
                        -alpha,
                        ply + 1,
                        allow_null=True,
                    )

                # PVS re-search when the null-window probe improved alpha but
                # did not already fail high.
                if alpha < score < beta:
                    score = -negamax(
                        board,
                        depth - 1,
                        -beta,
                        -alpha,
                        ply + 1,
                        allow_null=True,
                    )

        finally:
            board.pop()

        searched_moves += 1

        if score > best_score:
            best_score = score
            best_move = move

        if score > alpha:
            alpha = score

        if alpha >= beta:
            record_quiet_cutoff(board, move, depth, ply)
            break

    # At least the first legal move is always searched, so best_move should be
    # set even when frontier futility skipped later quiet moves.
    if best_move is None:
        best_move = moves[0]
        best_score = alpha

    if best_score <= alpha_start:
        flag = UPPER
    elif best_score >= beta_start:
        flag = LOWER
    else:
        flag = EXACT

    if len(TT) >= MAX_TT_SIZE:
        TT.clear()

    TT[key] = (
        depth,
        score_to_tt(best_score, ply),
        flag,
        best_move,
    )

    return best_score


# ============================================================
# ROOT SEARCH + ASPIRATION WINDOWS
# ============================================================


def search_root(board, depth, alpha, beta):
    alpha_start = alpha
    beta_start = beta

    entry = TT.get(position_key(board))
    tt_move = entry[3] if entry is not None else None

    moves = ordered_moves(board, tt_move, 0)

    if not moves:
        if board.is_check():
            return None, -MATE
        return None, 0

    best_score = -INF
    best_move = moves[0]
    searched_moves = 0

    for move_index, move in enumerate(moves):
        check_time(force=True)

        is_capture = board.is_capture(move)
        is_promotion = move.promotion is not None
        quiet = not is_capture and not is_promotion
        gives_check = board.gives_check(move)

        board.push(move)
        try:
            if searched_moves == 0:
                score = -negamax(
                    board,
                    depth - 1,
                    -beta,
                    -alpha,
                    1,
                    allow_null=True,
                )
            else:
                # Keep root moves at full nominal depth. LMR is applied inside
                # the tree, but root accuracy is worth the extra work.
                score = -negamax(
                    board,
                    depth - 1,
                    -alpha - 1,
                    -alpha,
                    1,
                    allow_null=True,
                )

                if alpha < score < beta:
                    score = -negamax(
                        board,
                        depth - 1,
                        -beta,
                        -alpha,
                        1,
                        allow_null=True,
                    )
        finally:
            board.pop()

        searched_moves += 1

        if score > best_score:
            best_score = score
            best_move = move

        if score > alpha:
            alpha = score

        if alpha >= beta:
            break

    # Save the root result too, primarily to improve ordering at the next
    # iterative-deepening depth.
    if best_score <= alpha_start:
        flag = UPPER
    elif best_score >= beta_start:
        flag = LOWER
    else:
        flag = EXACT

    TT[position_key(board)] = (
        depth,
        score_to_tt(best_score, 0),
        flag,
        best_move,
    )

    return best_move, best_score


def aspiration_search(board, depth, previous_score):
    if previous_score is None or depth < 4 or abs(previous_score) >= MATE_THRESHOLD:
        return search_root(board, depth, -INF, INF)

    window = 50

    while True:
        alpha = max(-INF, previous_score - window)
        beta = min(INF, previous_score + window)

        move, score = search_root(board, depth, alpha, beta)

        if score <= alpha:
            window *= 2
        elif score >= beta:
            window *= 2
        else:
            return move, score

        if window >= 1600:
            return search_root(board, depth, -INF, INF)


# ============================================================
# ENTRYPOINT
# ============================================================


def get_move(fen: str, time_left_ms: int) -> str:
    global NODES
    global DEADLINE

    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        raise ValueError("get_move called on terminal position")

    # Keep useful game-to-game-in-process TT information, but reset the
    # ply-dependent killer table each move. History is aged instead of erased.
    KILLERS.clear()
    decay_history()

    if len(TT) > MAX_TT_SIZE:
        TT.clear()
    if len(EVAL_CACHE) > MAX_EVAL_CACHE_SIZE:
        EVAL_CACHE.clear()

    budget_ms = choose_time_budget(time_left_ms)
    DEADLINE = time.monotonic() + budget_ms / 1000.0
    NODES = 0

    # Guaranteed legal emergency fallback.
    best_move = legal_moves[0]
    best_score = None
    depth = 1

    while True:
        start_nodes = NODES
        start_time = time.monotonic()

        try:
            move, score = aspiration_search(board, depth, best_score)
        except SearchTimeout:
            break

        if move is None:
            break

        # Only commit results from a fully completed iteration.
        best_move = move
        best_score = score

        elapsed = time.monotonic() - start_time
        depth_nodes = NODES - start_nodes

        print(
            f"depth={depth} score={best_score} "
            f"nodes={depth_nodes} total_nodes={NODES} "
            f"time={elapsed:.3f}s"
        )

        # No reason to spend time deepening a proven mate beyond its distance.
        if abs(best_score) >= MATE_THRESHOLD:
            break

        depth += 1

    # Defensive legality check. This should never fire, but a legal fallback is
    # preferable to losing a tournament game if an internal bug slips through.
    if best_move not in board.legal_moves:
        best_move = legal_moves[0]

    print(
        f"playing={best_move.uci()} completed_depth={depth - 1} "
        f"total_nodes={NODES} nn={NN_AVAILABLE}"
    )

    return best_move.uci()

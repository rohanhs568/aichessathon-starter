"""Verify exported V1 weights and the tournament agent locally.

Run from the repository root after exporting weights/v1.npz and replacing
agent.py:

    python training/verify_v1_agent.py \
        --checkpoint training/models/v1_natural_600m/final.pt

Checks:
    * agent actually loaded the learned weights
    * exported/Numba inference matches the PyTorch training network
    * random test positions evaluate consistently
    * get_move returns legal moves on representative positions
    * rough uncached evaluator throughput
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
import time
from pathlib import Path

# When this file is executed as `python training/verify_v1_agent.py`, Python
# puts `training/` rather than the repository root at sys.path[0]. Add the
# repository root explicitly so the tournament `agent.py` can be imported.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import chess
import torch

import agent


def load_training_module(path: Path):
    spec = importlib.util.spec_from_file_location("train_v1_for_verify", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def torch_raw_cp(model, train_v1, fen: str, k_cp: float) -> int:
    encoded = train_v1.encode_fen(fen)
    if encoded is None:
        raise RuntimeError(f"Training encoder rejected FEN: {fen}")

    stm_ids, opp_ids, piece_count, _material, _stm = encoded

    stm = torch.tensor(
        [train_v1.padded_feature_row(stm_ids)],
        dtype=torch.long,
    )
    opp = torch.tensor(
        [train_v1.padded_feature_row(opp_ids)],
        dtype=torch.long,
    )

    with torch.no_grad():
        stm_hidden = model.embedding(stm).sum(dim=1)
        opp_hidden = model.embedding(opp).sum(dim=1)
        stm_hidden = model.screlu(stm_hidden + model.hidden_bias)
        opp_hidden = model.screlu(opp_hidden + model.hidden_bias)
        combined = torch.cat((stm_hidden, opp_hidden), dim=1)
        raw_outputs = model.output(combined)

        if model.output_buckets == 1:
            bucket = 0
        else:
            bucket = max(
                0,
                min(model.output_buckets - 1, (piece_count - 2) // 4),
            )

        raw = float(raw_outputs[0, bucket])

    cp = max(-4000.0, min(4000.0, raw * k_cp))
    return int(round(cp))


def random_positions(count: int, seed: int):
    rng = random.Random(seed)
    board = chess.Board()
    positions = []

    while len(positions) < count:
        if board.is_game_over() or len(board.move_stack) >= 100:
            board = chess.Board()

        moves = list(board.legal_moves)
        if not moves:
            board = chess.Board()
            continue

        board.push(rng.choice(moves))
        if board.king(chess.WHITE) is not None and board.king(chess.BLACK) is not None:
            positions.append(board.fen())

    return positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=250)
    args = parser.parse_args()

    if not agent.NN_AVAILABLE:
        raise SystemExit(
            "FAIL: agent did not load weights/v1.npz. "
            f"Reason: {agent.NN_ERROR}"
        )

    train_v1 = load_training_module(Path("training/train_v1.py"))
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint.get("config", {})

    hidden = int(config.get("hidden", 64))
    buckets = int(config.get("buckets", 8))
    k_cp = float(config.get("k_cp", 400.0))

    model = train_v1.V1Evaluator(hidden, buckets)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    positions = random_positions(args.positions, seed=20260904)

    max_error = 0
    total_error = 0

    # Avoid cache hiding mistakes in the comparison.
    agent.EVAL_CACHE.clear()

    for fen in positions:
        board = chess.Board(fen)
        agent_score = agent.evaluate(board)
        torch_score = torch_raw_cp(model, train_v1, fen, k_cp)
        error = abs(agent_score - torch_score)
        max_error = max(max_error, error)
        total_error += error

        if error > 1:
            raise SystemExit(
                "FAIL: exported inference differs from PyTorch\n"
                f"FEN: {fen}\n"
                f"agent={agent_score}, torch={torch_score}, error={error}"
            )

    print(
        f"PASS inference equivalence: {len(positions)} positions, "
        f"max error={max_error} cp, "
        f"mean error={total_error / len(positions):.3f} cp"
    )

    # Uncached evaluation throughput. Use distinct positions and clear the
    # cache first so this measures real feature extraction + network inference.
    bench_positions = positions * 20
    boards = [chess.Board(fen) for fen in bench_positions]
    agent.EVAL_CACHE.clear()

    start = time.perf_counter()
    for board in boards:
        # Duplicate positions eventually hit cache. Clear periodically to keep
        # the result conservative and close to uncached inference cost.
        if len(agent.EVAL_CACHE) > 200:
            agent.EVAL_CACHE.clear()
        agent.evaluate(board)
    elapsed = time.perf_counter() - start

    print(
        f"Evaluator benchmark: {len(boards) / elapsed:,.0f} evals/s "
        f"over {len(boards):,} calls ({elapsed:.3f}s)"
    )

    smoke_fens = [
        chess.STARTING_FEN,
        "r1bq1rk1/ppp2ppp/2np1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 8",
        "r3r1k1/pp1n1ppp/2pb4/8/3P4/2N1PN2/PPQ2PPP/2RR2K1 w - - 2 17",
        "8/5pk1/6p1/3P3p/4P2P/5KP1/8/8 w - - 0 42",
    ]

    for index, fen in enumerate(smoke_fens, start=1):
        board = chess.Board(fen)
        start = time.perf_counter()
        move_text = agent.get_move(fen, 5_000)
        elapsed = time.perf_counter() - start
        move = chess.Move.from_uci(move_text)

        if move not in board.legal_moves:
            raise SystemExit(
                f"FAIL legal-move smoke {index}: {move_text} is illegal in {fen}"
            )

        print(
            f"PASS move {index}: {move_text} legal, "
            f"wall={elapsed:.3f}s"
        )

    print("ALL V1 AGENT CHECKS PASSED")


if __name__ == "__main__":
    main()

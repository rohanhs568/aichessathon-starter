"""Compare runtime calibration and piece sensitivity of V1 checkpoints."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import chess
import numpy as np
import torch

from training import train_v1
from training.train_v1_linear_recalibrate import (
    LinearV1Evaluator,
    evaluate_linear,
    load_linear_validation,
)


@torch.no_grad()
def model_cp(model, fens, device, k_cp):
    rows = []

    for fen in fens:
        encoded = train_v1.encode_fen(fen)
        if encoded is None:
            rows.append(np.nan)
            continue

        stm, opp, count, _material, _side = encoded

        stm_t = torch.from_numpy(
            np.expand_dims(train_v1.padded_feature_row(stm), 0)
        ).to(device=device, dtype=torch.long)
        opp_t = torch.from_numpy(
            np.expand_dims(train_v1.padded_feature_row(opp), 0)
        ).to(device=device, dtype=torch.long)
        cnt_t = torch.tensor([count], device=device, dtype=torch.long)

        rows.append(float(model(stm_t, opp_t, cnt_t).item() * k_cp))

    return np.asarray(rows)


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    model = LinearV1Evaluator(
        hidden_size=int(config.get("hidden", 64)),
        output_buckets=int(config.get("buckets", 8)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return checkpoint, model


def piece_sensitivity(model, validation_csv, device, k_cp, max_positions=400):
    values = defaultdict(list)
    sampled = 0

    with validation_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if sampled >= max_positions:
                break

            try:
                board = chess.Board(row["fen"])
            except ValueError:
                continue

            if board.is_check():
                continue

            side = board.turn
            original = model_cp(model, [board.fen()], device, k_cp)[0]
            if not np.isfinite(original):
                continue

            used = False

            for piece_type, name in [
                (chess.PAWN, "pawn"),
                (chess.KNIGHT, "knight"),
                (chess.BISHOP, "bishop"),
                (chess.ROOK, "rook"),
                (chess.QUEEN, "queen"),
            ]:
                squares = list(board.pieces(piece_type, side))
                if not squares:
                    continue

                altered = board.copy(stack=False)
                altered.remove_piece_at(squares[0])

                changed = model_cp(
                    model,
                    [altered.fen()],
                    device,
                    k_cp,
                )[0]

                if np.isfinite(changed):
                    values[name].append(original - changed)
                    used = True

            if used:
                sampled += 1

    return {
        name: float(np.median(vals))
        for name, vals in values.items()
        if vals
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("training/data/samples/lichess_validation_250k.csv"),
    )
    parser.add_argument("--piece-sample", type=int, default=400)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    for path in args.checkpoints:
        checkpoint, model = load_model(path, device)
        config = checkpoint.get("config", {})
        k_cp = float(config.get("k_cp", 400.0))
        cp_clip = float(config.get("cp_clip", 2000.0))

        validation = load_linear_validation(
            args.validation,
            k_cp,
            cp_clip,
        )
        metrics = evaluate_linear(
            model,
            validation,
            device,
            k_cp,
            cp_clip,
            16384,
        )
        sensitivity = piece_sensitivity(
            model,
            args.validation,
            device,
            k_cp,
            args.piece_sample,
        )

        print()
        print("=" * 78)
        print(path)
        print(f"objective: {config.get('objective', 'legacy_tanh_training')}")

        for key, value in metrics.items():
            print(f"{key:30s}: {value:.4f}")

        print("piece-removal median implied values:")
        for name in ("pawn", "knight", "bishop", "rook", "queen"):
            print(f"  {name:8s}: {sensitivity.get(name, float('nan')):8.2f}")

        bishop = sensitivity.get("bishop")
        rook = sensitivity.get("rook")
        if bishop not in (None, 0.0) and rook is not None:
            print(f"  rook/bishop ratio: {rook / bishop:.3f}")


if __name__ == "__main__":
    main()

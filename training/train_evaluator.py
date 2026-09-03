import argparse
import csv
import random
from pathlib import Path

import chess
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


INPUT_SIZE = 768


def encode_fen(fen):
    """
    Encode a position from the side-to-move perspective.

    Features:
        2 relative colours
        x 6 piece types
        x 64 squares
        = 768

    Channels 0..383: side-to-move's pieces
    Channels 384..767: opponent's pieces

    When Black is to move, mirror the board vertically so that
    both colours are represented from the same perspective.
    """

    parts = fen.split()

    # Lichess evaluation dump uses 4-field FENs.
    # python-chess can be kept happy by supplying move counters.
    if len(parts) == 4:
        fen = fen + " 0 1"

    board = chess.Board(fen)
    side_to_move = board.turn

    features = np.zeros(INPUT_SIZE, dtype=np.uint8)

    for square, piece in board.piece_map().items():

        # 0 = our piece, 1 = opponent piece
        relative_colour = 0 if piece.color == side_to_move else 1

        # From Black's perspective, flip ranks:
        # a8 -> a1, e7 -> e2, etc.
        canonical_square = (
            square
            if side_to_move == chess.WHITE
            else chess.square_mirror(square)
        )

        piece_index = piece.piece_type - 1

        index = (
            relative_colour * 6 * 64
            + piece_index * 64
            + canonical_square
        )

        features[index] = 1

    return features


def load_dataset(path, clip_cp):
    features = []
    targets = []

    skipped_mates = 0
    clipped = 0

    with path.open(newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            target = row["target_cp_stm"]

            # Mate positions currently have no ordinary CP target.
            if target == "":
                skipped_mates += 1
                continue

            cp = float(target)

            clipped_cp = max(-clip_cp, min(clip_cp, cp))

            if clipped_cp != cp:
                clipped += 1

            features.append(encode_fen(row["fen"]))
            targets.append(clipped_cp)

    X = np.stack(features)
    y = np.asarray(targets, dtype=np.float32)

    print(f"Usable positions: {len(y):,}")
    print(f"Mate rows skipped: {skipped_mates:,}")
    print(f"CP values clipped: {clipped:,}")
    print(f"Feature matrix: {X.shape}")

    return X, y


class Evaluator(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def evaluate_model(model, loader, device, scale):
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device=device, dtype=torch.float32)
            y_batch = y_batch.to(device)

            prediction = model(X_batch)

            predictions.append(prediction.cpu())
            targets.append(y_batch.cpu())

    predictions = torch.cat(predictions) * scale
    targets = torch.cat(targets) * scale

    error = predictions - targets

    mae = error.abs().mean().item()
    rmse = torch.sqrt((error ** 2).mean()).item()

    # Positions near 0 are noisy for sign classification,
    # so measure sign accuracy only when teacher score has
    # a meaningful preference.
    decisive = targets.abs() >= 50

    if decisive.any():
        sign_accuracy = (
            torch.sign(predictions[decisive])
            == torch.sign(targets[decisive])
        ).float().mean().item()
    else:
        sign_accuracy = float("nan")

    within_50 = (error.abs() <= 50).float().mean().item()
    within_100 = (error.abs() <= 100).float().mean().item()
    within_200 = (error.abs() <= 200).float().mean().item()

    return {
        "mae": mae,
        "rmse": rmse,
        "sign_accuracy": sign_accuracy,
        "within_50": within_50,
        "within_100": within_100,
        "within_200": within_200,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)

    parser.add_argument("--clip-cp", type=float, default=2000)
    parser.add_argument("--scale", type=float, default=400)

    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    X, y_cp = load_dataset(
        args.input,
        args.clip_cp,
    )

    # Normalize targets to make optimisation numerically easier.
    y = y_cp / args.scale

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(
        len(y),
        generator=generator,
    )

    validation_size = int(
        len(y) * args.validation_fraction
    )

    validation_indices = permutation[:validation_size]
    training_indices = permutation[validation_size:]

    train_dataset = TensorDataset(
        X[training_indices],
        y[training_indices],
    )

    validation_dataset = TensorDataset(
        X[validation_indices],
        y[validation_indices],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = Evaluator(args.hidden).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print(model)
    print(f"Parameters: {parameter_count:,}")
    print(f"Training positions: {len(train_dataset):,}")
    print(f"Validation positions: {len(validation_dataset):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    # Huber-style loss.
    #
    # With scale=400 and beta=0.25, errors within roughly
    # 100 centipawns receive quadratic loss, while large
    # errors are treated more robustly.
    loss_function = nn.SmoothL1Loss(beta=0.25)

    print()

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_positions = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(
                device=device,
                dtype=torch.float32,
            )

            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            prediction = model(X_batch)

            loss = loss_function(
                prediction,
                y_batch,
            )

            loss.backward()
            optimizer.step()

            batch_size = len(y_batch)

            total_loss += loss.item() * batch_size
            total_positions += batch_size

        training_loss = total_loss / total_positions

        metrics = evaluate_model(
            model,
            validation_loader,
            device,
            args.scale,
        )

        print(
            f"epoch={epoch:02d} "
            f"loss={training_loss:.4f} "
            f"val_mae={metrics['mae']:.1f}cp "
            f"val_rmse={metrics['rmse']:.1f}cp "
            f"sign={metrics['sign_accuracy']:.3f}"
        )

    final_metrics = evaluate_model(
        model,
        validation_loader,
        device,
        args.scale,
    )

    zero_baseline_mae = (
        y[validation_indices].abs().mean().item()
        * args.scale
    )

    print()
    print("Final validation")
    print("----------------")
    print(f"MAE:              {final_metrics['mae']:.1f} cp")
    print(f"RMSE:             {final_metrics['rmse']:.1f} cp")
    print(
        f"Sign accuracy:    "
        f"{100 * final_metrics['sign_accuracy']:.1f}%"
    )
    print(
        f"Within 50 cp:     "
        f"{100 * final_metrics['within_50']:.1f}%"
    )
    print(
        f"Within 100 cp:    "
        f"{100 * final_metrics['within_100']:.1f}%"
    )
    print(
        f"Within 200 cp:    "
        f"{100 * final_metrics['within_200']:.1f}%"
    )
    print(f"Zero-model MAE:   {zero_baseline_mae:.1f} cp")

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_size": INPUT_SIZE,
            "hidden_size": args.hidden,
            "scale": args.scale,
            "clip_cp": args.clip_cp,
            "perspective": "side_to_move",
            "encoding": "relative_piece_square_rank_mirror",
        },
        args.output,
    )

    print()
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
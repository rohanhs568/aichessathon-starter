import argparse
import csv
import json
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


# ============================================================
# V1 learned evaluator
# ============================================================
# Major features deliberately included together:
#   * dual perspective
#   * horizontal king mirroring
#   * castling-right features
#   * SCReLU hidden activation
#   * piece-count output buckets
#   * bounded tanh target/output
#   * optional broad data reweighting
#   * checkpoint/resume
#   * CUDA-safe AdamW (foreach=False)
#
# This is a new trainer. It does not replace the earlier baseline trainer.
# ============================================================

PIECE_INDEX = {
    "p": 0,
    "n": 1,
    "b": 2,
    "r": 3,
    "q": 4,
    "k": 5,
}

MATERIAL_VALUE = {
    "p": 100,
    "n": 320,
    "b": 330,
    "r": 500,
    "q": 900,
    "k": 0,
}

# Relative-colour piece-square features.
PIECE_SQUARE_FEATURES = 2 * 6 * 64  # 768

# Four relative castling features:
# own kingside, own queenside, opponent kingside, opponent queenside.
CASTLING_BASE = PIECE_SQUARE_FEATURES
CASTLING_FEATURES = 4
INPUT_FEATURES = PIECE_SQUARE_FEATURES + CASTLING_FEATURES  # 772

PADDING_INDEX = INPUT_FEATURES
MAX_STANDARD_PIECES = 32
MAX_ACTIVE_FEATURES = MAX_STANDARD_PIECES + CASTLING_FEATURES  # 36


def _parse_board(board_fen):
    """Return a list of (piece_char, rank, file), or None if malformed."""
    ranks = board_fen.split("/")
    if len(ranks) != 8:
        return None

    pieces = []

    for fen_rank_index, rank_text in enumerate(ranks):
        board_rank = 7 - fen_rank_index
        file_index = 0

        for char in rank_text:
            if char.isdigit():
                empty = int(char)
                if not 1 <= empty <= 8:
                    return None
                file_index += empty
                if file_index > 8:
                    return None
                continue

            piece_type = char.lower()
            if piece_type not in PIECE_INDEX or file_index >= 8:
                return None

            pieces.append((char, board_rank, file_index))
            file_index += 1

        if file_index != 8:
            return None

    if len(pieces) > MAX_STANDARD_PIECES:
        return None

    return pieces


def _perspective_features(pieces, castling, perspective_white):
    """
    Encode the position from one player's point of view.

    The perspective player's pieces are always "ours". Black perspective
    is vertically mirrored, so both sides conceptually move upward.

    Then the board is horizontally mirrored when the perspective king is
    on files e-h. This lets equivalent king-side/queen-side structures share
    weights and keeps the perspective king on files a-d.
    """
    own_king_file = None

    for char, _rank, file_index in pieces:
        if char.lower() != "k":
            continue

        piece_is_white = char.isupper()
        if piece_is_white == perspective_white:
            own_king_file = file_index
            break

    # Standard positions should contain the perspective king.
    if own_king_file is None:
        return None

    mirror_files = own_king_file >= 4
    indices = []

    for char, board_rank, file_index in pieces:
        piece_type = char.lower()
        piece_is_white = char.isupper()
        is_ours = piece_is_white == perspective_white
        relative_colour = 0 if is_ours else 1

        canonical_rank = (
            board_rank if perspective_white else 7 - board_rank
        )
        canonical_file = 7 - file_index if mirror_files else file_index
        square = canonical_rank * 8 + canonical_file

        feature_index = (
            relative_colour * 6 * 64
            + PIECE_INDEX[piece_type] * 64
            + square
        )

        if not 0 <= feature_index < PIECE_SQUARE_FEATURES:
            return None

        indices.append(feature_index)

    # Relative castling rights.
    if perspective_white:
        own_k = "K" in castling
        own_q = "Q" in castling
        opp_k = "k" in castling
        opp_q = "q" in castling
    else:
        own_k = "k" in castling
        own_q = "q" in castling
        opp_k = "K" in castling
        opp_q = "Q" in castling

    # Horizontal mirroring swaps king-side and queen-side geometry.
    if mirror_files:
        own_k, own_q = own_q, own_k
        opp_k, opp_q = opp_q, opp_k

    rights = (own_k, own_q, opp_k, opp_q)
    for offset, active in enumerate(rights):
        if active:
            indices.append(CASTLING_BASE + offset)

    if len(indices) > MAX_ACTIVE_FEATURES:
        return None

    return indices


def encode_fen(fen):
    """
    Encode a FEN for the V1 dual-perspective network.

    Returns:
        stm_indices
        opponent_indices
        piece_count
        material_stm
        side_to_move

    Invalid/non-standard positions return None.
    """
    fields = fen.split()
    if len(fields) < 3:
        return None

    board_fen, stm, castling = fields[0], fields[1], fields[2]
    if stm not in ("w", "b"):
        return None

    pieces = _parse_board(board_fen)
    if pieces is None:
        return None

    white_features = _perspective_features(
        pieces,
        castling,
        perspective_white=True,
    )
    black_features = _perspective_features(
        pieces,
        castling,
        perspective_white=False,
    )

    if white_features is None or black_features is None:
        return None

    white_material = 0
    black_material = 0

    for char, _rank, _file in pieces:
        value = MATERIAL_VALUE[char.lower()]
        if char.isupper():
            white_material += value
        else:
            black_material += value

    if stm == "w":
        stm_features = white_features
        opponent_features = black_features
        material_stm = white_material - black_material
    else:
        stm_features = black_features
        opponent_features = white_features
        material_stm = black_material - white_material

    return (
        stm_features,
        opponent_features,
        len(pieces),
        material_stm,
        stm,
    )


def teacher_cp_stm(side_to_move, cp_white):
    sign = 1.0 if side_to_move == "w" else -1.0
    return sign * cp_white


def target_from_label(side_to_move, cp_white, mate_white, k_cp):
    """
    Bounded target in [-1, 1].

        ordinary CP: tanh(cp_stm / K)
        forced mate: +/-1
    """
    sign = 1.0 if side_to_move == "w" else -1.0

    if cp_white is not None:
        cp_stm = sign * cp_white
        return math.tanh(cp_stm / k_cp), False, cp_stm

    if mate_white is not None:
        mate_stm = sign * mate_white
        if mate_stm > 0:
            return 1.0, True, None
        if mate_stm < 0:
            return -1.0, True, None
        return 0.0, True, None

    return None


def sample_weight(profile, piece_count, cp_stm, is_mate):
    """
    Broad data reweighting experiment.

    'natural' gives every training example equal weight.

    'rebalanced' uses moderate inverse-frequency-style weights based on the
    aggregate Lichess scan. It intentionally avoids extreme weights. This is
    not micro-tuning: it tests whether reducing domination by high-piece-count
    and near-zero-eval positions improves generalisation.
    """
    if profile == "natural":
        return 1.0

    if profile != "rebalanced":
        raise ValueError(f"Unknown data profile: {profile}")

    # Piece-count groups. Natural source is heavily weighted toward 24-32.
    if piece_count <= 7:
        phase_weight = 1.60
    elif piece_count <= 15:
        phase_weight = 1.15
    elif piece_count <= 23:
        phase_weight = 0.95
    else:
        phase_weight = 0.75

    if is_mate:
        eval_weight = 0.75
    else:
        absolute_cp = abs(cp_stm)
        if absolute_cp < 50:
            eval_weight = 0.70
        elif absolute_cp < 150:
            eval_weight = 1.00
        elif absolute_cp < 400:
            eval_weight = 1.15
        elif absolute_cp < 1000:
            eval_weight = 1.05
        else:
            eval_weight = 1.15

    return phase_weight * eval_weight


class V1Evaluator(nn.Module):
    """
    Dual-perspective NNUE-style evaluator.

    Each perspective uses the same feature embedding, so chess symmetries are
    shared. The two activated hidden vectors are concatenated, then a
    piece-count bucket selects the final output head.
    """

    def __init__(self, hidden_size=64, output_buckets=8):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_buckets = output_buckets

        self.embedding = nn.Embedding(
            INPUT_FEATURES + 1,
            hidden_size,
            padding_idx=PADDING_INDEX,
        )

        input_bound = 1.0 / math.sqrt(INPUT_FEATURES)

        with torch.no_grad():
            self.embedding.weight[:INPUT_FEATURES].uniform_(
                -input_bound,
                input_bound,
            )
            self.embedding.weight[PADDING_INDEX].zero_()

        self.hidden_bias = nn.Parameter(torch.empty(hidden_size))
        nn.init.uniform_(
            self.hidden_bias,
            -input_bound,
            input_bound,
        )

        self.output = nn.Linear(
            2 * hidden_size,
            output_buckets,
        )

    @staticmethod
    def screlu(x):
        return torch.clamp(x, 0.0, 1.0).square()

    def forward(self, stm_features, opponent_features, piece_counts):
        stm_hidden = self.embedding(stm_features).sum(dim=1)
        opp_hidden = self.embedding(opponent_features).sum(dim=1)

        stm_hidden = self.screlu(stm_hidden + self.hidden_bias)
        opp_hidden = self.screlu(opp_hidden + self.hidden_bias)

        combined = torch.cat((stm_hidden, opp_hidden), dim=1)
        raw_outputs = self.output(combined)

        if self.output_buckets == 1:
            raw = raw_outputs[:, 0]
        else:
            bucket = torch.div(
                piece_counts - 2,
                4,
                rounding_mode="floor",
            )
            bucket = torch.clamp(
                bucket,
                0,
                self.output_buckets - 1,
            )
            raw = raw_outputs.gather(
                1,
                bucket.unsqueeze(1),
            ).squeeze(1)

        # The model's actual chess value is bounded.
        return torch.tanh(raw)


def padded_feature_row(indices):
    row = np.full(
        MAX_ACTIVE_FEATURES,
        PADDING_INDEX,
        dtype=np.int16,
    )
    row[: len(indices)] = indices
    return row


def load_validation(path, k_cp):
    print("Loading permanent validation set...", flush=True)

    stm_rows = []
    opp_rows = []
    piece_counts = []
    targets = []
    cp_stm_values = []
    material_scores = []
    mate_flags = []

    skipped_nonstandard = 0

    with path.open(newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            encoded = encode_fen(row["fen"])
            if encoded is None:
                skipped_nonstandard += 1
                continue

            stm_indices, opp_indices, piece_count, material, stm = encoded

            cp_white = None if row["cp_white"] == "" else float(row["cp_white"])
            mate_white = None if row["mate_white"] == "" else float(row["mate_white"])

            result = target_from_label(
                stm,
                cp_white,
                mate_white,
                k_cp,
            )
            if result is None:
                continue

            target, is_mate, cp_stm = result

            stm_rows.append(padded_feature_row(stm_indices))
            opp_rows.append(padded_feature_row(opp_indices))
            piece_counts.append(piece_count)
            targets.append(target)
            cp_stm_values.append(np.nan if cp_stm is None else cp_stm)
            material_scores.append(material)
            mate_flags.append(is_mate)

    validation = (
        np.stack(stm_rows),
        np.stack(opp_rows),
        np.asarray(piece_counts, dtype=np.int16),
        np.asarray(targets, dtype=np.float32),
        np.asarray(cp_stm_values, dtype=np.float32),
        np.asarray(material_scores, dtype=np.float32),
        np.asarray(mate_flags, dtype=bool),
    )

    print(f"Validation positions: {len(targets):,}", flush=True)
    print(f"Non-standard skipped: {skipped_nonstandard:,}", flush=True)
    print(f"Mate positions: {sum(mate_flags):,}", flush=True)

    return validation


def bounded_to_cp(values, k_cp):
    # Prevent atanh from exploding numerically at +/-1.
    values = np.clip(values, -0.999, 0.999)
    return k_cp * np.arctanh(values)


@torch.no_grad()
def evaluate_model(model, validation, device, k_cp, batch_size):
    (
        stm_X,
        opp_X,
        piece_counts,
        targets,
        cp_stm_values,
        material_scores,
        mate_flags,
    ) = validation

    model.eval()
    predictions = []

    for start in range(0, len(targets), batch_size):
        end = min(start + batch_size, len(targets))

        stm_tensor = torch.from_numpy(stm_X[start:end]).to(
            device=device,
            dtype=torch.long,
        )
        opp_tensor = torch.from_numpy(opp_X[start:end]).to(
            device=device,
            dtype=torch.long,
        )
        count_tensor = torch.from_numpy(piece_counts[start:end]).to(
            device=device,
            dtype=torch.long,
        )

        output = model(stm_tensor, opp_tensor, count_tensor)
        predictions.append(output.cpu().numpy())

    predictions = np.concatenate(predictions)

    target_error = predictions - targets
    cp_mask = ~mate_flags

    predicted_cp = bounded_to_cp(predictions[cp_mask], k_cp)
    teacher_cp = cp_stm_values[cp_mask]
    cp_error = predicted_cp - teacher_cp

    decisive_mask = np.abs(teacher_cp) >= 50
    balanced_mask = np.abs(teacher_cp) <= 400
    medium_mask = np.abs(teacher_cp) <= 800

    def safe_mean(values):
        return float(np.mean(values)) if len(values) else float("nan")

    metrics = {
        "target_mae": float(np.mean(np.abs(target_error))),
        "target_rmse": float(np.sqrt(np.mean(target_error ** 2))),
        "cp_mae": float(np.mean(np.abs(cp_error))),
        "cp_rmse": float(np.sqrt(np.mean(cp_error ** 2))),
        "balanced_cp_mae": safe_mean(np.abs(cp_error[balanced_mask])),
        "medium_cp_mae": safe_mean(np.abs(cp_error[medium_mask])),
        "sign_accuracy": safe_mean(
            np.sign(predicted_cp[decisive_mask])
            == np.sign(teacher_cp[decisive_mask])
        ),
        "within_50_balanced": safe_mean(
            np.abs(cp_error[balanced_mask]) <= 50
        ),
        "within_100_balanced": safe_mean(
            np.abs(cp_error[balanced_mask]) <= 100
        ),
        "mate_sign_accuracy": safe_mean(
            np.sign(predictions[mate_flags])
            == np.sign(targets[mate_flags])
        ),
        "material_balanced_mae": safe_mean(
            np.abs(
                material_scores[cp_mask][balanced_mask]
                - teacher_cp[balanced_mask]
            )
        ),
    }

    model.train()
    return metrics


def read_shard(path, shuffle_buffer, rng):
    process = subprocess.Popen(
        ["zstd", "-dc", str(path)],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1024 * 1024,
    )

    buffer = []

    try:
        for line in process.stdout:
            buffer.append(line)

            if len(buffer) >= shuffle_buffer:
                rng.shuffle(buffer)
                for item in buffer:
                    yield item
                buffer.clear()

        if buffer:
            rng.shuffle(buffer)
            for item in buffer:
                yield item

    finally:
        process.stdout.close()
        code = process.wait()

        # SIGPIPE is expected when training deliberately stops mid-shard.
        if code not in (0, -13):
            raise RuntimeError(
                f"zstd failed for {path} with exit code {code}"
            )


def parse_position_count(text):
    text = text.strip().lower()
    multiplier = 1

    if text.endswith("k"):
        multiplier = 1_000
        text = text[:-1]
    elif text.endswith("m"):
        multiplier = 1_000_000
        text = text[:-1]
    elif text.endswith("b"):
        multiplier = 1_000_000_000
        text = text[:-1]

    return int(float(text) * multiplier)


def format_position_count(value):
    if value >= 1_000_000_000:
        number = value / 1_000_000_000
        return f"{int(number)}b" if number.is_integer() else f"{number:g}b"
    if value >= 1_000_000:
        number = value / 1_000_000
        return f"{int(number)}m" if number.is_integer() else f"{number:g}m"
    if value >= 1_000:
        return f"{value // 1_000}k"
    return str(value)


def parse_checkpoint_list(text):
    return sorted(
        {
            parse_position_count(item.strip())
            for item in text.split(",")
            if item.strip()
        }
    )


def learning_rate_at_position(
    positions_seen,
    max_positions,
    start_lr,
    end_lr_factor,
):
    progress = min(positions_seen / max_positions, 1.0)
    return start_lr * (
        1.0 - progress * (1.0 - end_lr_factor)
    )


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def save_checkpoint(
    path,
    model,
    optimizer,
    config,
    positions_seen,
    metrics,
):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "positions_seen": positions_seen,
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def validate_resume_config(saved_config, config):
    # Resume only if the experiment definition is identical.
    keys = (
        "hidden",
        "buckets",
        "batch_size",
        "learning_rate",
        "end_lr_factor",
        "weight_decay",
        "k_cp",
        "min_depth",
        "max_positions",
        "data_profile",
    )

    for key in keys:
        if key not in saved_config:
            continue
        if saved_config[key] != config[key]:
            raise RuntimeError(
                f"Resume config mismatch for '{key}': "
                f"checkpoint={saved_config[key]}, current={config[key]}"
            )


def print_metrics(metrics):
    print(f"  Target MAE:       {metrics['target_mae']:.4f}")
    print(f"  CP MAE:           {metrics['cp_mae']:.1f} cp")
    print(f"  Balanced CP MAE:  {metrics['balanced_cp_mae']:.1f} cp")
    print(f"  Medium CP MAE:    {metrics['medium_cp_mae']:.1f} cp")
    print(f"  Sign accuracy:    {100 * metrics['sign_accuracy']:.1f}%")
    print(f"  Mate sign:        {100 * metrics['mate_sign_accuracy']:.1f}%")
    print(
        f"  <=50cp balanced:  "
        f"{100 * metrics['within_50_balanced']:.1f}%"
    )
    print(
        f"  <=100cp balanced: "
        f"{100 * metrics['within_100_balanced']:.1f}%"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)

    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--buckets", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--validation-batch-size", type=int, default=16384)
    parser.add_argument("--shuffle-buffer", type=int, default=65536)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--end-lr-factor", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--k-cp", type=float, default=400.0)
    parser.add_argument("--min-depth", type=int, default=0)
    parser.add_argument(
        "--data-profile",
        choices=("natural", "rebalanced"),
        default="natural",
    )

    parser.add_argument("--max-positions", type=str, default="10m")
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="1m,5m,10m,25m,50m,100m,250m",
    )
    parser.add_argument("--log-every", type=str, default="1m")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    max_positions = parse_position_count(args.max_positions)
    checkpoint_positions = [
        value
        for value in parse_checkpoint_list(args.checkpoints)
        if value <= max_positions
    ]
    log_every = parse_position_count(args.log_every)

    args.run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}",
            flush=True,
        )

    shard_paths = sorted(
        args.shards.glob("lichess_train_*.tsv.zst")
    )
    if not shard_paths:
        raise RuntimeError(f"No shards found in {args.shards}")

    print(f"Training shards: {len(shard_paths)}", flush=True)

    validation = load_validation(args.validation, args.k_cp)

    model = V1Evaluator(
        hidden_size=args.hidden,
        output_buckets=args.buckets,
    ).to(device)

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    print(f"Parameters: {parameter_count:,}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        foreach=False,  # more stable on this WSL/Pascal CUDA setup
    )

    config = {
        "model": "v1_dual_perspective_king_mirror_castling_screlu_tanh",
        "hidden": args.hidden,
        "buckets": args.buckets,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "end_lr_factor": args.end_lr_factor,
        "weight_decay": args.weight_decay,
        "k_cp": args.k_cp,
        "min_depth": args.min_depth,
        "data_profile": args.data_profile,
        "max_positions": max_positions,
        "seed": args.seed,
        "input_features": INPUT_FEATURES,
        "max_active_features": MAX_ACTIVE_FEATURES,
    }

    positions_seen = 0

    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {args.resume}"
            )

        print(f"Resuming from: {args.resume}", flush=True)
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )

        validate_resume_config(
            checkpoint.get("config", {}),
            config,
        )

        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        positions_seen = int(checkpoint["positions_seen"])

        if positions_seen >= max_positions:
            raise RuntimeError(
                f"Checkpoint already has {positions_seen:,} positions, "
                f"max is {max_positions:,}."
            )

        resumed_lr = learning_rate_at_position(
            positions_seen,
            max_positions,
            args.learning_rate,
            args.end_lr_factor,
        )
        set_optimizer_lr(optimizer, resumed_lr)

        print(f"Resumed at: {positions_seen:,} positions", flush=True)
        print(f"Resumed LR: {resumed_lr:.2e}", flush=True)

    with (args.run_dir / "config.json").open("w") as file:
        json.dump(config, file, indent=2)

    results_path = args.run_dir / "results.jsonl"

    metrics = evaluate_model(
        model,
        validation,
        device,
        args.k_cp,
        args.validation_batch_size,
    )

    print()
    print(
        "Resume validation:" if positions_seen else "Initial validation:"
    )
    print_metrics(metrics)
    print()

    remaining_checkpoints = [
        value
        for value in checkpoint_positions
        if value > positions_seen
    ]

    next_log = ((positions_seen // log_every) + 1) * log_every

    rng_seed = args.seed if positions_seen == 0 else args.seed + positions_seen
    rng = random.Random(rng_seed)

    session_start_positions = positions_seen
    start_time = time.time()
    epoch = 0

    batch_stm = np.full(
        (args.batch_size, MAX_ACTIVE_FEATURES),
        PADDING_INDEX,
        dtype=np.int16,
    )
    batch_opp = np.full(
        (args.batch_size, MAX_ACTIVE_FEATURES),
        PADDING_INDEX,
        dtype=np.int16,
    )
    batch_counts = np.empty(args.batch_size, dtype=np.int16)
    batch_targets = np.empty(args.batch_size, dtype=np.float32)
    batch_weights = np.empty(args.batch_size, dtype=np.float32)

    batch_index = 0
    total_loss = 0.0
    total_loss_positions = 0

    while positions_seen < max_positions:
        epoch += 1
        epoch_shards = list(shard_paths)
        rng.shuffle(epoch_shards)

        print(
            f"Starting data epoch {epoch} "
            f"(global positions={positions_seen:,})",
            flush=True,
        )

        for shard_path in epoch_shards:
            for line in read_shard(
                shard_path,
                args.shuffle_buffer,
                rng,
            ):
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 6:
                    continue

                (
                    fen,
                    cp_text,
                    mate_text,
                    depth_text,
                    _knodes_text,
                    piece_count_text,
                ) = parts

                try:
                    depth = int(depth_text)
                    metadata_piece_count = int(piece_count_text)
                except ValueError:
                    continue

                if depth < args.min_depth:
                    continue
                if metadata_piece_count > MAX_STANDARD_PIECES:
                    continue

                encoded = encode_fen(fen)
                if encoded is None:
                    continue

                stm_indices, opp_indices, piece_count, _material, stm = encoded
                if piece_count != metadata_piece_count:
                    continue

                cp_white = None if cp_text == "" else float(cp_text)
                mate_white = None if mate_text == "" else float(mate_text)

                target_result = target_from_label(
                    stm,
                    cp_white,
                    mate_white,
                    args.k_cp,
                )
                if target_result is None:
                    continue

                target, is_mate, cp_stm = target_result

                batch_stm[batch_index].fill(PADDING_INDEX)
                batch_opp[batch_index].fill(PADDING_INDEX)

                batch_stm[batch_index, : len(stm_indices)] = stm_indices
                batch_opp[batch_index, : len(opp_indices)] = opp_indices
                batch_counts[batch_index] = piece_count
                batch_targets[batch_index] = target
                batch_weights[batch_index] = sample_weight(
                    args.data_profile,
                    piece_count,
                    cp_stm,
                    is_mate,
                )

                batch_index += 1

                if (
                    batch_index < args.batch_size
                    and positions_seen + batch_index < max_positions
                ):
                    continue

                effective_batch = batch_index

                stm_view = batch_stm[:effective_batch]
                opp_view = batch_opp[:effective_batch]
                count_view = batch_counts[:effective_batch]

                if (
                    stm_view.min() < 0
                    or stm_view.max() > PADDING_INDEX
                    or opp_view.min() < 0
                    or opp_view.max() > PADDING_INDEX
                ):
                    raise RuntimeError(
                        "Invalid feature index detected before CUDA transfer"
                    )

                if (
                    count_view.min() < 2
                    or count_view.max() > MAX_STANDARD_PIECES
                ):
                    raise RuntimeError(
                        "Invalid piece count detected before CUDA transfer"
                    )

                stm_tensor = torch.from_numpy(stm_view).to(
                    device=device,
                    dtype=torch.long,
                )
                opp_tensor = torch.from_numpy(opp_view).to(
                    device=device,
                    dtype=torch.long,
                )
                count_tensor = torch.from_numpy(count_view).to(
                    device=device,
                    dtype=torch.long,
                )
                target_tensor = torch.from_numpy(
                    batch_targets[:effective_batch]
                ).to(
                    device=device,
                    dtype=torch.float32,
                )
                weight_tensor = torch.from_numpy(
                    batch_weights[:effective_batch]
                ).to(
                    device=device,
                    dtype=torch.float32,
                )

                # Keep the average gradient scale comparable across profiles.
                weight_tensor = weight_tensor / weight_tensor.mean()

                optimizer.zero_grad(set_to_none=True)

                predictions = model(
                    stm_tensor,
                    opp_tensor,
                    count_tensor,
                )

                squared_error = (predictions - target_tensor).square()
                loss = (squared_error * weight_tensor).mean()

                loss.backward()
                optimizer.step()

                positions_seen += effective_batch
                total_loss += loss.item() * effective_batch
                total_loss_positions += effective_batch

                current_lr = learning_rate_at_position(
                    positions_seen,
                    max_positions,
                    args.learning_rate,
                    args.end_lr_factor,
                )
                set_optimizer_lr(optimizer, current_lr)

                batch_index = 0

                if positions_seen >= next_log:
                    elapsed = time.time() - start_time
                    session_positions = positions_seen - session_start_positions
                    throughput = session_positions / elapsed
                    average_loss = total_loss / total_loss_positions

                    memory_mb = 0.0
                    if device.type == "cuda":
                        memory_mb = (
                            torch.cuda.max_memory_allocated() / 1024**2
                        )

                    print(
                        f"seen={positions_seen:,} "
                        f"loss={average_loss:.5f} "
                        f"lr={current_lr:.2e} "
                        f"rate={throughput:,.0f}/s "
                        f"gpu_mem={memory_mb:.0f}MB "
                        f"elapsed={elapsed / 60:.1f}m",
                        flush=True,
                    )

                    total_loss = 0.0
                    total_loss_positions = 0

                    while next_log <= positions_seen:
                        next_log += log_every

                while (
                    remaining_checkpoints
                    and positions_seen >= remaining_checkpoints[0]
                ):
                    checkpoint_value = remaining_checkpoints.pop(0)

                    checkpoint_metrics = evaluate_model(
                        model,
                        validation,
                        device,
                        args.k_cp,
                        args.validation_batch_size,
                    )

                    result = {
                        "checkpoint": checkpoint_value,
                        "positions_seen": positions_seen,
                        "elapsed_seconds": time.time() - start_time,
                        **checkpoint_metrics,
                    }

                    print()
                    print(
                        f"Checkpoint {format_position_count(checkpoint_value)}"
                    )
                    print_metrics(checkpoint_metrics)
                    print()

                    with results_path.open("a") as file:
                        file.write(json.dumps(result) + "\n")

                    checkpoint_path = (
                        args.run_dir
                        / f"checkpoint_{format_position_count(checkpoint_value)}.pt"
                    )

                    save_checkpoint(
                        checkpoint_path,
                        model,
                        optimizer,
                        config,
                        positions_seen,
                        checkpoint_metrics,
                    )

                    print(f"Saved {checkpoint_path}", flush=True)

                if positions_seen >= max_positions:
                    break

            if positions_seen >= max_positions:
                break

    final_metrics = evaluate_model(
        model,
        validation,
        device,
        args.k_cp,
        args.validation_batch_size,
    )

    final_path = args.run_dir / "final.pt"
    save_checkpoint(
        final_path,
        model,
        optimizer,
        config,
        positions_seen,
        final_metrics,
    )

    elapsed = time.time() - start_time
    session_positions = positions_seen - session_start_positions
    throughput = session_positions / elapsed

    print()
    print("=" * 60)
    print("V1 TRAINING COMPLETE")
    print("=" * 60)
    print(f"Positions seen: {positions_seen:,}")
    print(f"This session:   {session_positions:,}")
    print(f"Runtime:        {elapsed / 60:.2f} min")
    print(f"Throughput:     {throughput:,.0f} positions/s")
    print_metrics(final_metrics)
    print(f"Saved final model: {final_path}")


if __name__ == "__main__":
    main()

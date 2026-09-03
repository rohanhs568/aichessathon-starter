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


# ------------------------------------------------------------
# Feature representation
# ------------------------------------------------------------

# 2 relative colours x 6 piece types x 64 squares
INPUT_FEATURES = 768

# Padding value used to fill unused piece slots.
# The embedding table has one extra row for this.
PADDING_INDEX = 768

MAX_STANDARD_PIECES = 32

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


# ------------------------------------------------------------
# Fast FEN encoding
# ------------------------------------------------------------

def encode_fen(fen):
    """
    Convert a FEN into up to 32 active feature indices.

    Representation is relative to side to move.

    Returns:
        indices
        piece_count
        material_stm

    Invalid or non-standard positions return None.
    """

    fields = fen.split()

    if len(fields) < 2:
        return None

    board_fen = fields[0]
    stm = fields[1]

    if stm not in ("w", "b"):
        return None

    ranks = board_fen.split("/")

    # A valid chess board must have exactly 8 ranks.
    if len(ranks) != 8:
        return None

    white_to_move = stm == "w"

    indices = []

    own_material = 0
    opponent_material = 0

    for fen_rank_index, rank_text in enumerate(ranks):

        # FEN starts at rank 8 and moves down to rank 1.
        board_rank = 7 - fen_rank_index

        file_index = 0

        for char in rank_text:

            if char.isdigit():

                empty_squares = int(char)

                # Individual FEN digit must represent 1..8 empties.
                if not 1 <= empty_squares <= 8:
                    return None

                file_index += empty_squares

                if file_index > 8:
                    return None

                continue

            piece_type = char.lower()

            if piece_type not in PIECE_INDEX:
                return None

            # A piece cannot occur beyond the h-file.
            if file_index >= 8:
                return None

            piece_is_white = char.isupper()

            is_ours = (
                piece_is_white
                if white_to_move
                else not piece_is_white
            )

            relative_colour = (
                0 if is_ours else 1
            )

            # Mirror ranks when Black is to move.
            canonical_rank = (
                board_rank
                if white_to_move
                else 7 - board_rank
            )

            square = (
                canonical_rank * 8
                + file_index
            )

            if not 0 <= square < 64:
                return None

            feature_index = (
                relative_colour * 6 * 64
                + PIECE_INDEX[piece_type] * 64
                + square
            )

            # Defensive check before this ever reaches CUDA.
            if not 0 <= feature_index < INPUT_FEATURES:
                return None

            indices.append(feature_index)

            value = MATERIAL_VALUE[piece_type]

            if is_ours:
                own_material += value
            else:
                opponent_material += value

            file_index += 1

        # Every FEN rank must describe exactly 8 squares.
        if file_index != 8:
            return None

    # Ordinary chess positions cannot contain >32 pieces.
    if len(indices) > MAX_STANDARD_PIECES:
        return None

    return (
        indices,
        len(indices),
        own_material - opponent_material,
    )


# ------------------------------------------------------------
# Target construction
# ------------------------------------------------------------

def make_target(
    side_to_move,
    cp_white,
    mate_white,
    clip_cp,
    mate_cp,
):
    """
    Convert Lichess White-POV labels to side-to-move POV.

    Ordinary evaluations are clipped.

    Mate positions are retained and mapped to +/- mate_cp.

    This is deliberately configurable because mate/target
    handling is one of our experiments.
    """

    sign = 1.0 if side_to_move == "w" else -1.0

    if cp_white is not None:
        cp_stm = sign * cp_white

        cp_stm = max(
            -clip_cp,
            min(clip_cp, cp_stm),
        )

        return float(cp_stm), False

    if mate_white is not None:

        mate_stm = sign * mate_white

        if mate_stm > 0:
            return float(mate_cp), True

        if mate_stm < 0:
            return float(-mate_cp), True

        # Extremely unusual mate=0 record.
        return 0.0, True

    return None


# ------------------------------------------------------------
# Network
# ------------------------------------------------------------

class SparseEvaluator(nn.Module):

    def __init__(
        self,
        hidden_size=64,
        activation="relu",
        output_buckets=1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.activation_name = activation
        self.output_buckets = output_buckets

        # Each active chess feature has one hidden vector.
        #
        # Summing these is mathematically equivalent to:
        #
        #     Linear(768, hidden)
        #
        # applied to the sparse binary board representation.
        self.embedding = nn.Embedding(
            INPUT_FEATURES + 1,
            hidden_size,
            padding_idx=PADDING_INDEX,
        )

        # Match the scale used by nn.Linear(768, hidden).
        #
        # Our embedding rows are exactly the columns of the equivalent
        # dense linear layer, so they should have the same initialization
        # scale rather than nn.Embedding's much larger default.
        input_bound = 1.0 / math.sqrt(INPUT_FEATURES)

        with torch.no_grad():
            self.embedding.weight[
                :INPUT_FEATURES
            ].uniform_(
                -input_bound,
                input_bound,
            )

            self.embedding.weight[
                PADDING_INDEX
            ].zero_()

        self.hidden_bias = nn.Parameter(
            torch.empty(hidden_size)
        )

        nn.init.uniform_(
            self.hidden_bias,
            -input_bound,
            input_bound,
        )

        self.output = nn.Linear(
            hidden_size,
            output_buckets,
        )

    def activate(self, x):

        if self.activation_name == "relu":
            return torch.relu(x)

        if self.activation_name == "screlu":
            # Squared clipped ReLU
            return torch.clamp(x, 0.0, 1.0).square()

        raise ValueError(
            f"Unknown activation: {self.activation_name}"
        )

    def forward(
        self,
        feature_indices,
        piece_counts,
    ):
        # [batch, 32, hidden]
        embeddings = self.embedding(feature_indices)

        # Equivalent to W x for sparse binary x.
        hidden = embeddings.sum(dim=1)

        hidden = hidden + self.hidden_bias

        hidden = self.activate(hidden)

        outputs = self.output(hidden)

        if self.output_buckets == 1:
            return outputs[:, 0]

        # Piece-count bucket scheme:
        #
        # 3-5 pieces   -> bucket 0
        # 6-9          -> bucket 1
        # ...
        #
        # With 8 buckets and standard chess this spans
        # the full 3-32 piece range.
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

        return outputs.gather(
            1,
            bucket.unsqueeze(1),
        ).squeeze(1)


# ------------------------------------------------------------
# Validation data
# ------------------------------------------------------------

def load_validation(
    path,
    clip_cp,
    mate_cp,
):

    print(
        "Loading permanent validation set...",
        flush=True,
    )

    feature_rows = []
    piece_counts = []
    targets = []
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

            indices, piece_count, material = encoded


            cp_white = (
                None
                if row["cp_white"] == ""
                else float(row["cp_white"])
            )

            mate_white = (
                None
                if row["mate_white"] == ""
                else float(row["mate_white"])
            )

            target_result = make_target(
                row["side_to_move"],
                cp_white,
                mate_white,
                clip_cp,
                mate_cp,
            )

            if target_result is None:
                continue

            target, is_mate = target_result

            feature_row = np.full(
                MAX_STANDARD_PIECES,
                PADDING_INDEX,
                dtype=np.int16,
            )

            feature_row[:len(indices)] = indices

            feature_rows.append(feature_row)

            piece_counts.append(piece_count)
            targets.append(target)
            material_scores.append(material)
            mate_flags.append(is_mate)

    X = np.stack(feature_rows)

    piece_counts = np.asarray(
        piece_counts,
        dtype=np.int16,
    )

    targets = np.asarray(
        targets,
        dtype=np.float32,
    )

    material_scores = np.asarray(
        material_scores,
        dtype=np.float32,
    )

    mate_flags = np.asarray(
        mate_flags,
        dtype=bool,
    )

    print(
        f"Validation positions: {len(targets):,}",
        flush=True,
    )

    print(
        f"Non-standard skipped: {skipped_nonstandard:,}",
        flush=True,
    )

    print(
        f"Mate positions: {mate_flags.sum():,}",
        flush=True,
    )

    return (
        X,
        piece_counts,
        targets,
        material_scores,
        mate_flags,
    )


# ------------------------------------------------------------
# Validation metrics
# ------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model,
    validation,
    device,
    scale,
    batch_size,
):

    (
        X,
        piece_counts,
        targets,
        material_scores,
        mate_flags,
    ) = validation

    model.eval()

    predictions = []

    for start in range(
        0,
        len(targets),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(targets),
        )

        features = torch.from_numpy(
            X[start:end]
        ).to(
            device=device,
            dtype=torch.long,
        )

        counts = torch.from_numpy(
            piece_counts[start:end]
        ).to(
            device=device,
            dtype=torch.long,
        )

        output = model(
            features,
            counts,
        )

        predictions.append(
            output.cpu().numpy()
        )

    predictions = np.concatenate(
        predictions
    )

    predictions_cp = predictions * scale

    error = predictions_cp - targets

    all_mae = np.mean(
        np.abs(error)
    )

    all_rmse = np.sqrt(
        np.mean(error ** 2)
    )

    cp_mask = ~mate_flags

    cp_error = (
    predictions_cp[cp_mask]
    - targets[cp_mask]
    )

    cp_mae = np.mean(
        np.abs(cp_error)
    )

    cp_rmse = np.sqrt(
        np.mean(cp_error ** 2)
    )

    cp_zero_mae = np.mean(
        np.abs(targets[cp_mask])
    )

    cp_decisive = (
        cp_mask
        & (np.abs(targets) >= 50)
    )

    sign_accuracy = np.mean(
        np.sign(
            predictions_cp[cp_decisive]
        )
        ==
        np.sign(
            targets[cp_decisive]
        )
    )
    if mate_flags.any():

        mate_sign_accuracy = np.mean(
            np.sign(
                predictions_cp[mate_flags]
            )
            ==
            np.sign(
                targets[mate_flags]
            )
        )

    else:
        mate_sign_accuracy = float("nan")

    zero_mae = np.mean(
        np.abs(targets)
    )

    # Compare the handcrafted material evaluator only
    # on ordinary CP-labelled positions.
    material_mae = np.mean(
        np.abs(
            material_scores[cp_mask]
            - targets[cp_mask]
        )
    )

    within_50 = np.mean(
        np.abs(error[cp_mask]) <= 50
    )

    within_100 = np.mean(
        np.abs(error[cp_mask]) <= 100
    )

    model.train()

    return {
        "all_mae": float(all_mae),
        "all_rmse": float(all_rmse),
        "cp_mae": float(cp_mae),
        "sign_accuracy": float(sign_accuracy),
        "mate_sign_accuracy": float(
            mate_sign_accuracy
        ),
        "within_50": float(within_50),
        "within_100": float(within_100),
        "zero_mae": float(zero_mae),
        "material_mae": float(material_mae),
        "cp_rmse": float(cp_rmse),
        "cp_zero_mae": float(cp_zero_mae),
    }


# ------------------------------------------------------------
# Shard streaming
# ------------------------------------------------------------

def read_shard(
    path,
    shuffle_buffer,
    rng,
):
    """
    Stream one compressed TSV shard.

    Lines are locally shuffled in reasonably large blocks.

    Full global shuffling of 400m positions would be expensive,
    so we combine:
        - random shard order
        - random order within large local buffers

    This gives good stochasticity without materialising the
    full dataset.
    """

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

        # -13 is SIGPIPE. It is expected when training reaches
# max_positions part-way through a shard and deliberately
# stops consuming the decompression stream.
        if code not in (0, -13):
            raise RuntimeError(
                f"zstd failed for {path} "
                f"with exit code {code}"
            )

# ------------------------------------------------------------
# Checkpoints
# ------------------------------------------------------------

def format_position_count(value):

    if value >= 1_000_000_000:
        number = value / 1_000_000_000

        if number.is_integer():
            return f"{int(number)}b"

        return f"{number:g}b"

    if value >= 1_000_000:
        number = value / 1_000_000

        if number.is_integer():
            return f"{int(number)}m"

        return f"{number:g}m"

    if value >= 1_000:
        return f"{value // 1000}k"

    return str(value)


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


def parse_checkpoint_list(text):

    values = []

    for item in text.split(","):
        item = item.strip()

        if item:
            values.append(
                parse_position_count(item)
            )

    return sorted(set(values))


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
            "optimizer_state_dict":
                optimizer.state_dict(),
            "positions_seen": positions_seen,
            "metrics": metrics,
            "config": config,
        },
        path,
    )


# ------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shards",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--validation",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--hidden",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--activation",
        choices=["relu", "screlu"],
        default="relu",
    )

    parser.add_argument(
        "--buckets",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16384,
    )

    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=16384,
    )

    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=65536,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--end-lr-factor",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--clip-cp",
        type=float,
        default=2000.0,
    )

    parser.add_argument(
        "--mate-cp",
        type=float,
        default=2000.0,
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=400.0,
    )

    parser.add_argument(
        "--min-depth",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max-positions",
        type=str,
        default="1m",
    )

    parser.add_argument(
        "--checkpoints",
        type=str,
        default="1m,5m,10m,25m,50m,100m,200m,400m",
    )

    parser.add_argument(
        "--log-every",
        type=str,
        default="250k",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    max_positions = parse_position_count(
        args.max_positions
    )

    checkpoint_positions = [
        value
        for value
        in parse_checkpoint_list(
            args.checkpoints
        )
        if value <= max_positions
    ]

    log_every = parse_position_count(
        args.log_every
    )

    args.run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}",
        flush=True,
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}",
            flush=True,
        )

    shard_paths = sorted(
        args.shards.glob(
            "lichess_train_*.tsv.zst"
        )
    )

    if not shard_paths:
        raise RuntimeError(
            f"No shards found in {args.shards}"
        )

    print(
        f"Training shards: {len(shard_paths)}",
        flush=True,
    )

    validation = load_validation(
        args.validation,
        args.clip_cp,
        args.mate_cp,
    )

    model = SparseEvaluator(
        hidden_size=args.hidden,
        activation=args.activation,
        output_buckets=args.buckets,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Parameters: {parameter_count:,}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Robust to very large evaluation mistakes.
    loss_function = nn.SmoothL1Loss(
        beta=0.25
    )

    config = {
        "hidden": args.hidden,
        "activation": args.activation,
        "buckets": args.buckets,
        "batch_size": args.batch_size,
        "learning_rate":
            args.learning_rate,
        "end_lr_factor":
            args.end_lr_factor,
        "weight_decay":
            args.weight_decay,
        "clip_cp": args.clip_cp,
        "mate_cp": args.mate_cp,
        "scale": args.scale,
        "min_depth": args.min_depth,
        "max_positions":
            max_positions,
        "seed": args.seed,
        "encoding":
            "sparse_relative_piece_square_rank_mirror",
    }

    with (
        args.run_dir / "config.json"
    ).open("w") as file:

        json.dump(
            config,
            file,
            indent=2,
        )

    results_path = (
        args.run_dir / "results.jsonl"
    )

    # Initial validation gives us the random-network
    # reference and validates the data path.
    initial_metrics = evaluate_model(
        model,
        validation,
        device,
        args.scale,
        args.validation_batch_size,
    )

    print()
    print(
        "Initial validation:",
        initial_metrics,
        flush=True,
    )
    print()

    positions_seen = 0
    epoch = 0
    next_log = log_every

    remaining_checkpoints = list(
        checkpoint_positions
    )

    start_time = time.time()

    # Preallocate CPU batches.
    batch_features = np.full(
        (
            args.batch_size,
            MAX_STANDARD_PIECES,
        ),
        PADDING_INDEX,
        dtype=np.int16,
    )

    batch_counts = np.empty(
        args.batch_size,
        dtype=np.int16,
    )

    batch_targets = np.empty(
        args.batch_size,
        dtype=np.float32,
    )

    batch_index = 0

    total_loss = 0.0
    total_loss_positions = 0

    rng = random.Random(args.seed)

    while positions_seen < max_positions:

        epoch += 1

        epoch_shards = list(
            shard_paths
        )

        rng.shuffle(
            epoch_shards
        )

        print(
            f"Starting data epoch {epoch}",
            flush=True,
        )

        for shard_path in epoch_shards:

            for line in read_shard(
                shard_path,
                args.shuffle_buffer,
                rng,
            ):

                parts = line.rstrip(
                    "\n"
                ).split("\t")

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
                    metadata_piece_count = int(
                        piece_count_text
                    )
                except ValueError:
                    continue

                if depth < args.min_depth:
                    continue

                # Ordinary standard chess cannot have
                # more than 32 pieces.
                if metadata_piece_count > 32:
                    continue

                encoded = encode_fen(fen)

                if encoded is None:
                    continue

                (
                    indices,
                    piece_count,
                    _material,
                ) = encoded

                # Shard metadata and parsed FEN should agree.
                if piece_count != metadata_piece_count:
                    continue

                fields = fen.split()

                side_to_move = fields[1]

                cp_white = (
                    None
                    if cp_text == ""
                    else float(cp_text)
                )

                mate_white = (
                    None
                    if mate_text == ""
                    else float(mate_text)
                )

                target_result = make_target(
                    side_to_move,
                    cp_white,
                    mate_white,
                    args.clip_cp,
                    args.mate_cp,
                )

                if target_result is None:
                    continue

                target_cp, _is_mate = (
                    target_result
                )

                # Reset row to padding.
                batch_features[
                    batch_index
                ].fill(PADDING_INDEX)

                batch_features[
                    batch_index,
                    :len(indices)
                ] = indices

                batch_counts[
                    batch_index
                ] = piece_count

                batch_targets[
                    batch_index
                ] = (
                    target_cp
                    / args.scale
                )

                batch_index += 1

                if (
                    batch_index
                    < args.batch_size
                    and
                    positions_seen
                    + batch_index
                    < max_positions
                ):
                    continue

                effective_batch = (
                    batch_index
                )

                # Never allow a bad embedding index to reach CUDA.
                feature_view = batch_features[
                    :effective_batch
                ]

                if (
                    feature_view.min() < 0
                    or feature_view.max() > PADDING_INDEX
                ):
                    raise RuntimeError(
                        "Invalid feature index detected "
                        "before CUDA transfer"
                    )

                count_view = batch_counts[
                    :effective_batch
                ]

                if (
                    count_view.min() < 1
                    or count_view.max() > MAX_STANDARD_PIECES
                ):
                    raise RuntimeError(
                        "Invalid piece count detected "
                        "before CUDA transfer"
                    )

                features_tensor = (
                    torch.from_numpy(
                        batch_features[
                            :effective_batch
                        ]
                    )
                    .to(
                        device=device,
                        dtype=torch.long,
                    )
                )

                counts_tensor = (
                    torch.from_numpy(
                        batch_counts[
                            :effective_batch
                        ]
                    )
                    .to(
                        device=device,
                        dtype=torch.long,
                    )
                )

                targets_tensor = (
                    torch.from_numpy(
                        batch_targets[
                            :effective_batch
                        ]
                    )
                    .to(
                        device=device,
                        dtype=torch.float32,
                    )
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                predictions = model(
                    features_tensor,
                    counts_tensor,
                )

                loss = loss_function(
                    predictions,
                    targets_tensor,
                )

                loss.backward()

                optimizer.step()

                positions_seen += (
                    effective_batch
                )

                total_loss += (
                    loss.item()
                    * effective_batch
                )

                total_loss_positions += (
                    effective_batch
                )

                # Linear learning-rate decay.
                progress = min(
                    positions_seen
                    / max_positions,
                    1.0,
                )

                lr_factor = (
                    1.0
                    - progress
                    * (
                        1.0
                        - args.end_lr_factor
                    )
                )

                current_lr = (
                    args.learning_rate
                    * lr_factor
                )

                for group in (
                    optimizer.param_groups
                ):
                    group["lr"] = current_lr

                batch_index = 0

                # ------------------------------------
                # Periodic logging
                # ------------------------------------

                if positions_seen >= next_log:

                    elapsed = (
                        time.time()
                        - start_time
                    )

                    throughput = (
                        positions_seen
                        / elapsed
                    )

                    average_loss = (
                        total_loss
                        / total_loss_positions
                    )

                    memory_mb = 0.0

                    if (
                        device.type
                        == "cuda"
                    ):
                        memory_mb = (
                            torch.cuda
                            .max_memory_allocated()
                            / 1024**2
                        )

                    print(
                        f"seen="
                        f"{positions_seen:,} "
                        f"loss="
                        f"{average_loss:.4f} "
                        f"lr="
                        f"{current_lr:.2e} "
                        f"rate="
                        f"{throughput:,.0f}/s "
                        f"gpu_mem="
                        f"{memory_mb:.0f}MB "
                        f"elapsed="
                        f"{elapsed / 60:.1f}m",
                        flush=True,
                    )

                    total_loss = 0.0
                    total_loss_positions = 0

                    while (
                        next_log
                        <= positions_seen
                    ):
                        next_log += (
                            log_every
                        )

                # ------------------------------------
                # Validation checkpoints
                # ------------------------------------

                while (
                    remaining_checkpoints
                    and
                    positions_seen
                    >= remaining_checkpoints[0]
                ):

                    checkpoint_value = (
                        remaining_checkpoints
                        .pop(0)
                    )

                    metrics = evaluate_model(
                        model,
                        validation,
                        device,
                        args.scale,
                        args.validation_batch_size,
                    )

                    elapsed = (
                        time.time()
                        - start_time
                    )

                    result = {
                        "checkpoint":
                            checkpoint_value,
                        "positions_seen":
                            positions_seen,
                        "elapsed_seconds":
                            elapsed,
                        **metrics,
                    }

                    print()
                    print(
                        f"Checkpoint "
                        f"{format_position_count(checkpoint_value)}",
                        flush=True,
                    )

                    print(
                        f"  CP MAE:       "
                        f"{metrics['cp_mae']:.1f} cp",
                        flush=True,
                    )

                    print(
                        f"  All MAE:      "
                        f"{metrics['all_mae']:.1f} cp",
                        flush=True,
                    )

                    print(
                        f"  Sign accuracy:"
                        f" "
                        f"{100 * metrics['sign_accuracy']:.1f}%",
                        flush=True,
                    )

                    print(
                        f"  Mate sign:    "
                        f"{100 * metrics['mate_sign_accuracy']:.1f}%",
                        flush=True,
                    )

                    print(
                        f"  Within 50cp:  "
                        f"{100 * metrics['within_50']:.1f}%",
                        flush=True,
                    )

                    print(
                        f"  Material MAE: "
                        f"{metrics['material_mae']:.1f} cp",
                        flush=True,
                    )

                    print()

                    with results_path.open(
                        "a"
                    ) as file:

                        file.write(
                            json.dumps(result)
                            + "\n"
                        )

                    checkpoint_path = (
                        args.run_dir
                        /
                        (
                            "checkpoint_"
                            + format_position_count(
                                checkpoint_value
                            )
                            + ".pt"
                        )
                    )

                    save_checkpoint(
                        checkpoint_path,
                        model,
                        optimizer,
                        config,
                        positions_seen,
                        metrics,
                    )

                    print(
                        f"Saved "
                        f"{checkpoint_path}",
                        flush=True,
                    )

                if (
                    positions_seen
                    >= max_positions
                ):
                    break

            if (
                positions_seen
                >= max_positions
            ):
                break

    # --------------------------------------------------------
    # Final model
    # --------------------------------------------------------

    final_metrics = evaluate_model(
        model,
        validation,
        device,
        args.scale,
        args.validation_batch_size,
    )

    final_path = (
        args.run_dir / "final.pt"
    )

    save_checkpoint(
        final_path,
        model,
        optimizer,
        config,
        positions_seen,
        final_metrics,
    )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Positions seen: "
        f"{positions_seen:,}"
    )

    print(
        f"Runtime: "
        f"{elapsed / 60:.2f} min"
    )

    print(
        f"Average throughput: "
        f"{positions_seen / elapsed:,.0f} positions/s"
    )

    print(
        f"Final CP MAE: "
        f"{final_metrics['cp_mae']:.1f} cp"
    )

    print(
        f"Final sign accuracy: "
        f"{100 * final_metrics['sign_accuracy']:.1f}%"
    )

    print(
        f"Final mate sign: "
        f"{100 * final_metrics['mate_sign_accuracy']:.1f}%"
    )

    print(
        f"Saved final model: "
        f"{final_path}"
    )


if __name__ == "__main__":
    main()
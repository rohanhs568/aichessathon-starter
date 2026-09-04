"""Continue the exact V1 architecture on the Stockfish/Lichess corpus.

This is the Stockfish-only long-run control experiment. It starts from an
existing V1 checkpoint, keeps the model architecture and optimizer state, and
continues training with a fresh low learning-rate decay schedule.

It deliberately does NOT mix LC0 data. The Stockfish-only continuation is the
control needed to tell whether later LC0 training adds value beyond simply
showing V1 more examples.

Example:
    python -m training.train_v1_continue_sf \
      --shards /mnt/d/ChessData/lichess_train_shards \
      --validation training/data/samples/lichess_validation_250k.csv \
      --init training/models/v1_natural_600m/final.pt \
      --run-dir training/models/v1_sf_long_1800m \
      --additional-positions 1.2b \
      --checkpoint-every 100m \
      --device auto
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from training import train_v1


def parse_count(text: str) -> int:
    return train_v1.parse_position_count(text)


def format_count(value: int) -> str:
    return train_v1.format_position_count(value)


def move_optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    base_config: dict,
    continuation_config: dict,
    global_positions_seen: int,
    session_positions_seen: int,
    metrics: dict,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "positions_seen": global_positions_seen,
            "session_positions_seen": session_positions_seen,
            "metrics": metrics,
            "config": base_config,
            "continuation_config": continuation_config,
        },
        path,
    )


def choose_device(text: str) -> torch.device:
    if text == "cpu":
        return torch.device("cpu")

    if text == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)

    parser.add_argument(
        "--additional-positions",
        type=str,
        default="1.2b",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=str,
        default="100m",
    )
    parser.add_argument(
        "--log-every",
        type=str,
        default="10m",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--start-lr",
        type=float,
        default=None,
        help=(
            "Continuation start LR. Default: LR stored in the input "
            "checkpoint optimizer, normally ~1e-4 at 600m."
        ),
    )
    parser.add_argument(
        "--end-lr-factor",
        type=float,
        default=0.25,
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
    parser.add_argument("--seed", type=int, default=20260904)

    args = parser.parse_args()

    if not args.init.is_file():
        raise FileNotFoundError(args.init)

    shard_paths = sorted(
        args.shards.glob("lichess_train_*.tsv.zst")
    )
    if not shard_paths:
        raise RuntimeError(f"No shards found in {args.shards}")

    additional_positions = parse_count(args.additional_positions)
    checkpoint_every = parse_count(args.checkpoint_every)
    log_every = parse_count(args.log_every)

    if additional_positions <= 0:
        raise ValueError("additional positions must be positive")

    device = choose_device(args.device)

    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}",
            flush=True,
        )
        print(
            f"CUDA capability: {torch.cuda.get_device_capability(0)}",
            flush=True,
        )

    print(f"Training shards: {len(shard_paths)}", flush=True)

    checkpoint = torch.load(
        args.init,
        map_location=device,
        weights_only=False,
    )

    base_config = dict(checkpoint.get("config", {}))

    hidden = int(base_config.get("hidden", 64))
    buckets = int(base_config.get("buckets", 8))
    batch_size = int(base_config.get("batch_size", 16384))
    weight_decay = float(base_config.get("weight_decay", 1e-5))
    k_cp = float(base_config.get("k_cp", 400.0))
    min_depth = int(base_config.get("min_depth", 0))
    data_profile = str(base_config.get("data_profile", "natural"))

    if data_profile != "natural":
        print(
            f"warning: continuing saved data_profile={data_profile}",
            flush=True,
        )

    global_start = int(checkpoint.get("positions_seen", 0))
    if global_start <= 0:
        raise RuntimeError(
            "input checkpoint has no positive positions_seen"
        )

    global_end = global_start + additional_positions

    print(
        f"Starting global position: {global_start:,}",
        flush=True,
    )
    print(
        f"Additional positions: {additional_positions:,}",
        flush=True,
    )
    print(
        f"Target global position: {global_end:,}",
        flush=True,
    )
    print(
        f"Batch size: {batch_size:,}",
        flush=True,
    )

    model = train_v1.V1Evaluator(
        hidden_size=hidden,
        output_buckets=buckets,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])

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
        lr=1e-4,
        weight_decay=weight_decay,
        foreach=False,
    )

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        move_optimizer_to_device(
            optimizer,
            device,
        )
        print(
            "Loaded AdamW optimizer moments from checkpoint.",
            flush=True,
        )
    else:
        print(
            "warning: no optimizer state found; starting fresh AdamW state",
            flush=True,
        )

    saved_lr = float(
        optimizer.param_groups[0].get("lr", 1e-4)
    )
    start_lr = (
        saved_lr
        if args.start_lr is None
        else float(args.start_lr)
    )
    end_lr = start_lr * args.end_lr_factor

    train_v1.set_optimizer_lr(
        optimizer,
        start_lr,
    )

    print(
        f"Continuation LR: {start_lr:.2e} -> {end_lr:.2e}",
        flush=True,
    )

    validation = train_v1.load_validation(
        args.validation,
        k_cp,
    )

    baseline_metrics = train_v1.evaluate_model(
        model,
        validation,
        device,
        k_cp,
        args.validation_batch_size,
    )

    print()
    print("Initial continuation validation:")
    train_v1.print_metrics(
        baseline_metrics
    )
    print()

    args.run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    continuation_config = {
        "source": "stockfish_lichess_only",
        "init_checkpoint": str(args.init),
        "global_start": global_start,
        "global_end": global_end,
        "additional_positions": additional_positions,
        "checkpoint_every": checkpoint_every,
        "batch_size": batch_size,
        "start_lr": start_lr,
        "end_lr": end_lr,
        "weight_decay": weight_decay,
        "k_cp": k_cp,
        "min_depth": min_depth,
        "data_profile": data_profile,
        "seed": args.seed,
        "device": str(device),
    }

    (
        args.run_dir / "continuation_config.json"
    ).write_text(
        json.dumps(
            continuation_config,
            indent=2,
        ),
        encoding="utf-8",
    )

    results_path = (
        args.run_dir
        / "results.jsonl"
    )

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    batch_stm = np.full(
        (
            batch_size,
            train_v1.MAX_ACTIVE_FEATURES,
        ),
        train_v1.PADDING_INDEX,
        dtype=np.int16,
    )
    batch_opp = np.full(
        (
            batch_size,
            train_v1.MAX_ACTIVE_FEATURES,
        ),
        train_v1.PADDING_INDEX,
        dtype=np.int16,
    )
    batch_counts = np.empty(
        batch_size,
        dtype=np.int16,
    )
    batch_targets = np.empty(
        batch_size,
        dtype=np.float32,
    )
    batch_weights = np.empty(
        batch_size,
        dtype=np.float32,
    )

    session_seen = 0
    batch_index = 0
    epoch = 0

    next_log = log_every
    next_checkpoint_global = (
        ((global_start // checkpoint_every) + 1)
        * checkpoint_every
    )

    total_loss = 0.0
    total_loss_positions = 0
    start_time = time.time()

    def continuation_lr(
        session_positions: int,
    ) -> float:
        progress = min(
            session_positions
            / additional_positions,
            1.0,
        )
        return (
            start_lr
            + progress
            * (end_lr - start_lr)
        )

    while session_seen < additional_positions:
        epoch += 1
        epoch_shards = list(shard_paths)
        rng.shuffle(epoch_shards)

        print(
            f"Starting continuation data epoch {epoch} "
            f"(global={global_start + session_seen:,})",
            flush=True,
        )

        for shard_path in epoch_shards:
            for line in train_v1.read_shard(
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
                    metadata_piece_count = int(
                        piece_count_text
                    )
                except ValueError:
                    continue

                if depth < min_depth:
                    continue

                if (
                    metadata_piece_count
                    > train_v1.MAX_STANDARD_PIECES
                ):
                    continue

                encoded = train_v1.encode_fen(
                    fen
                )
                if encoded is None:
                    continue

                (
                    stm_indices,
                    opp_indices,
                    piece_count,
                    _material,
                    stm,
                ) = encoded

                if piece_count != metadata_piece_count:
                    continue

                try:
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
                except ValueError:
                    continue

                target_result = (
                    train_v1.target_from_label(
                        stm,
                        cp_white,
                        mate_white,
                        k_cp,
                    )
                )

                if target_result is None:
                    continue

                target, is_mate, cp_stm = (
                    target_result
                )

                batch_stm[
                    batch_index
                ].fill(
                    train_v1.PADDING_INDEX
                )
                batch_opp[
                    batch_index
                ].fill(
                    train_v1.PADDING_INDEX
                )

                batch_stm[
                    batch_index,
                    : len(stm_indices),
                ] = stm_indices

                batch_opp[
                    batch_index,
                    : len(opp_indices),
                ] = opp_indices

                batch_counts[
                    batch_index
                ] = piece_count

                batch_targets[
                    batch_index
                ] = target

                batch_weights[
                    batch_index
                ] = train_v1.sample_weight(
                    data_profile,
                    piece_count,
                    cp_stm,
                    is_mate,
                )

                batch_index += 1

                if (
                    batch_index < batch_size
                    and session_seen + batch_index
                    < additional_positions
                ):
                    continue

                effective_batch = batch_index

                stm_tensor = torch.from_numpy(
                    batch_stm[:effective_batch]
                ).to(
                    device=device,
                    dtype=torch.long,
                )

                opp_tensor = torch.from_numpy(
                    batch_opp[:effective_batch]
                ).to(
                    device=device,
                    dtype=torch.long,
                )

                count_tensor = torch.from_numpy(
                    batch_counts[:effective_batch]
                ).to(
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

                weight_tensor = (
                    weight_tensor
                    / weight_tensor.mean()
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                predictions = model(
                    stm_tensor,
                    opp_tensor,
                    count_tensor,
                )

                squared_error = (
                    predictions
                    - target_tensor
                ).square()

                loss = (
                    squared_error
                    * weight_tensor
                ).mean()

                loss.backward()
                optimizer.step()

                session_seen += effective_batch
                global_seen = (
                    global_start
                    + session_seen
                )

                total_loss += (
                    loss.item()
                    * effective_batch
                )
                total_loss_positions += (
                    effective_batch
                )

                current_lr = continuation_lr(
                    session_seen
                )
                train_v1.set_optimizer_lr(
                    optimizer,
                    current_lr,
                )

                batch_index = 0

                if session_seen >= next_log:
                    elapsed = (
                        time.time()
                        - start_time
                    )
                    rate = (
                        session_seen
                        / elapsed
                    )
                    average_loss = (
                        total_loss
                        / total_loss_positions
                    )

                    gpu_memory = 0.0
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                        gpu_memory = (
                            torch.cuda.max_memory_allocated()
                            / 1024**2
                        )

                    print(
                        f"session={session_seen:,} "
                        f"global={global_seen:,} "
                        f"loss={average_loss:.5f} "
                        f"lr={current_lr:.2e} "
                        f"rate={rate:,.0f}/s "
                        f"gpu_mem={gpu_memory:.0f}MB "
                        f"elapsed={elapsed/3600:.2f}h",
                        flush=True,
                    )

                    total_loss = 0.0
                    total_loss_positions = 0

                    while next_log <= session_seen:
                        next_log += log_every

                while (
                    global_seen
                    >= next_checkpoint_global
                    and next_checkpoint_global
                    <= global_end
                ):
                    metrics = (
                        train_v1.evaluate_model(
                            model,
                            validation,
                            device,
                            k_cp,
                            args.validation_batch_size,
                        )
                    )

                    result = {
                        "checkpoint": (
                            next_checkpoint_global
                        ),
                        "positions_seen": global_seen,
                        "session_positions_seen": (
                            session_seen
                        ),
                        "elapsed_seconds": (
                            time.time()
                            - start_time
                        ),
                        "learning_rate": (
                            current_lr
                        ),
                        **metrics,
                    }

                    print()
                    print(
                        f"Checkpoint "
                        f"{format_count(next_checkpoint_global)}"
                    )
                    train_v1.print_metrics(
                        metrics
                    )
                    print()

                    with results_path.open(
                        "a",
                        encoding="utf-8",
                    ) as handle:
                        handle.write(
                            json.dumps(result)
                            + "\n"
                        )

                    checkpoint_path = (
                        args.run_dir
                        / (
                            "checkpoint_"
                            f"{format_count(next_checkpoint_global)}"
                            ".pt"
                        )
                    )

                    save_checkpoint(
                        checkpoint_path,
                        model,
                        optimizer,
                        base_config,
                        continuation_config,
                        global_seen,
                        session_seen,
                        metrics,
                    )

                    print(
                        f"Saved {checkpoint_path}",
                        flush=True,
                    )

                    next_checkpoint_global += (
                        checkpoint_every
                    )

                if (
                    session_seen
                    >= additional_positions
                ):
                    break

            if (
                session_seen
                >= additional_positions
            ):
                break

    final_metrics = (
        train_v1.evaluate_model(
            model,
            validation,
            device,
            k_cp,
            args.validation_batch_size,
        )
    )

    final_path = (
        args.run_dir
        / "final.pt"
    )

    save_checkpoint(
        final_path,
        model,
        optimizer,
        base_config,
        continuation_config,
        global_start + session_seen,
        session_seen,
        final_metrics,
    )

    elapsed = time.time() - start_time
    rate = session_seen / elapsed

    print()
    print("=" * 78)
    print("STOCKFISH-ONLY V1 CONTINUATION COMPLETE")
    print("=" * 78)
    print(
        f"Global positions: "
        f"{global_start + session_seen:,}"
    )
    print(
        f"Additional positions: "
        f"{session_seen:,}"
    )
    print(
        f"Elapsed: {elapsed/3600:.2f} h"
    )
    print(
        f"Average throughput: "
        f"{rate:,.0f} positions/s"
    )
    print(
        f"Final checkpoint: {final_path}"
    )
    print()
    train_v1.print_metrics(
        final_metrics
    )


if __name__ == "__main__":
    main()

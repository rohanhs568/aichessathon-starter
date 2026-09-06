"""Recalibrate the V1 evaluator directly in runtime CP units.

OLD objective:
    prediction = tanh(raw)
    target     = tanh(cp / K)
    loss       = MSE

NEW objective:
    prediction = raw
    target     = clip(cp, -CP_CLIP, CP_CLIP) / K
    loss       = Huber

The parameter names and architecture are unchanged, so the existing V1 exporter
and tournament runtime remain compatible: runtime score is still raw * K.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training import train_v1


class LinearV1Evaluator(train_v1.V1Evaluator):
    """Same parameters as V1Evaluator, but return raw output instead of tanh."""

    def forward(self, stm_features, opponent_features, piece_counts):
        stm_hidden = self.embedding(stm_features).sum(dim=1)
        opp_hidden = self.embedding(opponent_features).sum(dim=1)

        stm_hidden = self.screlu(stm_hidden + self.hidden_bias)
        opp_hidden = self.screlu(opp_hidden + self.hidden_bias)

        combined = torch.cat((stm_hidden, opp_hidden), dim=1)
        raw_outputs = self.output(combined)

        if self.output_buckets == 1:
            return raw_outputs[:, 0]

        bucket = torch.div(
            piece_counts - 2,
            4,
            rounding_mode="floor",
        )
        bucket = torch.clamp(bucket, 0, self.output_buckets - 1)
        return raw_outputs.gather(1, bucket.unsqueeze(1)).squeeze(1)


def parse_count(text: str) -> int:
    return train_v1.parse_position_count(text)


def format_count(value: int) -> str:
    return train_v1.format_position_count(value)


def choose_device(text: str) -> torch.device:
    if text == "cpu":
        return torch.device("cpu")
    if text == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linear_target(stm: str, cp_white, mate_white, k_cp: float, cp_clip: float):
    sign = 1.0 if stm == "w" else -1.0

    if cp_white is not None:
        cp_stm = sign * cp_white
        clipped = max(-cp_clip, min(cp_clip, cp_stm))
        return clipped / k_cp, False, cp_stm

    if mate_white is not None:
        mate_stm = sign * mate_white
        if mate_stm > 0:
            return cp_clip / k_cp, True, None
        if mate_stm < 0:
            return -cp_clip / k_cp, True, None
        return 0.0, True, None

    return None


def load_linear_validation(path: Path, k_cp: float, cp_clip: float):
    stm_rows = []
    opp_rows = []
    counts = []
    teacher_cp = []
    mate_signs = []

    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            encoded = train_v1.encode_fen(row["fen"])
            if encoded is None:
                continue

            stm_indices, opp_indices, piece_count, _material, stm = encoded
            cp_white = None if row["cp_white"] == "" else float(row["cp_white"])
            mate_white = None if row["mate_white"] == "" else float(row["mate_white"])

            if cp_white is not None:
                cp_stm = train_v1.teacher_cp_stm(stm, cp_white)
                stm_rows.append(train_v1.padded_feature_row(stm_indices))
                opp_rows.append(train_v1.padded_feature_row(opp_indices))
                counts.append(piece_count)
                teacher_cp.append(cp_stm)
                mate_signs.append(0)
            elif mate_white is not None:
                sign = 1.0 if stm == "w" else -1.0
                mate_stm = sign * mate_white
                stm_rows.append(train_v1.padded_feature_row(stm_indices))
                opp_rows.append(train_v1.padded_feature_row(opp_indices))
                counts.append(piece_count)
                teacher_cp.append(np.nan)
                mate_signs.append(1 if mate_stm > 0 else -1 if mate_stm < 0 else 0)

    return (
        np.stack(stm_rows),
        np.stack(opp_rows),
        np.asarray(counts, dtype=np.int16),
        np.asarray(teacher_cp, dtype=np.float32),
        np.asarray(mate_signs, dtype=np.int8),
    )


@torch.no_grad()
def evaluate_linear(model, validation, device, k_cp: float, cp_clip: float, batch_size: int):
    stm_x, opp_x, counts, teacher_cp, mate_signs = validation

    outputs = []
    model.eval()

    for start in range(0, len(counts), batch_size):
        end = min(start + batch_size, len(counts))
        stm = torch.from_numpy(stm_x[start:end]).to(device=device, dtype=torch.long)
        opp = torch.from_numpy(opp_x[start:end]).to(device=device, dtype=torch.long)
        cnt = torch.from_numpy(counts[start:end]).to(device=device, dtype=torch.long)
        outputs.append(model(stm, opp, cnt).cpu().numpy())

    raw = np.concatenate(outputs)
    pred_cp = raw * k_cp

    cp_mask = np.isfinite(teacher_cp)
    t = teacher_cp[cp_mask]
    p = pred_cp[cp_mask]
    tc = np.clip(t, -cp_clip, cp_clip)

    balanced = np.abs(t) <= 400
    medium = np.abs(t) <= 800
    calibration = np.abs(t) <= 1000
    decisive = np.abs(t) >= 50

    def mean(x):
        return float(np.mean(x)) if len(x) else float("nan")

    denom = max(1.0, float(np.dot(t[calibration], t[calibration])))
    slope = float(np.dot(t[calibration], p[calibration]) / denom)
    corr = float(np.corrcoef(t[calibration], p[calibration])[0, 1])
    residual = p[calibration] - t[calibration]

    mate_mask = mate_signs != 0
    mate_acc = mean(np.sign(raw[mate_mask]) == mate_signs[mate_mask])

    metrics = {
        "clipped_cp_mae": mean(np.abs(p - tc)),
        "balanced_cp_mae": mean(np.abs(p[balanced] - t[balanced])),
        "medium_cp_mae": mean(np.abs(p[medium] - t[medium])),
        "sign_accuracy": mean(np.sign(p[decisive]) == np.sign(t[decisive])),
        "calibration_slope_abs1000": slope,
        "correlation_abs1000": corr,
        "residual_std_abs1000": float(np.std(residual)),
        "mate_sign_accuracy": mate_acc,
        "max_abs_prediction_cp": float(np.max(np.abs(p))),
    }

    model.train()
    return metrics


def print_metrics(label: str, metrics: dict) -> None:
    print(label, flush=True)
    for key, value in metrics.items():
        print(f"  {key:28s}: {value:.4f}", flush=True)


def save_checkpoint(
    path,
    model,
    optimizer,
    base_config,
    objective_config,
    global_positions,
    session_positions,
    metrics,
):
    config = dict(base_config)
    config["objective"] = "linear_huber_cp"
    config["cp_clip"] = objective_config["cp_clip"]
    config["huber_beta_cp"] = objective_config["huber_beta_cp"]

    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "positions_seen": global_positions,
            "linear_session_positions": session_positions,
            "metrics": metrics,
            "config": config,
            "linear_objective_config": objective_config,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)

    parser.add_argument("--additional-positions", default="100m")
    parser.add_argument("--checkpoint-every", default="25m")
    parser.add_argument("--log-every", default="5m")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--start-lr", type=float, default=5e-5)
    parser.add_argument("--end-lr-factor", type=float, default=0.2)
    parser.add_argument("--cp-clip", type=float, default=2000.0)
    parser.add_argument("--huber-beta-cp", type=float, default=200.0)
    parser.add_argument("--validation-batch-size", type=int, default=16384)
    parser.add_argument("--shuffle-buffer", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()

    if not args.init.is_file():
        raise FileNotFoundError(args.init)

    shards = sorted(args.shards.glob("lichess_train_*.tsv.zst"))
    if not shards:
        raise RuntimeError(f"No training shards in {args.shards}")

    device = choose_device(args.device)
    checkpoint = torch.load(args.init, map_location=device, weights_only=False)
    base_config = dict(checkpoint.get("config", {}))

    hidden = int(base_config.get("hidden", 64))
    buckets = int(base_config.get("buckets", 8))
    batch_size = int(base_config.get("batch_size", 16384))
    weight_decay = float(base_config.get("weight_decay", 1e-5))
    k_cp = float(base_config.get("k_cp", 400.0))
    min_depth = int(base_config.get("min_depth", 0))
    data_profile = str(base_config.get("data_profile", "natural"))

    additional = parse_count(args.additional_positions)
    checkpoint_every = parse_count(args.checkpoint_every)
    log_every = parse_count(args.log_every)

    global_start = int(checkpoint.get("positions_seen", 0))
    global_end = global_start + additional

    model = LinearV1Evaluator(hidden_size=hidden, output_buckets=buckets).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    # New objective, so use fresh optimizer moments.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.start_lr,
        weight_decay=weight_decay,
        foreach=False,
    )

    validation = load_linear_validation(args.validation, k_cp, args.cp_clip)

    print(f"Device: {device}", flush=True)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"Starting checkpoint positions: {global_start:,}", flush=True)
    print(f"Linear recalibration positions: {additional:,}", flush=True)
    print(f"CP clip: +/-{args.cp_clip:g}", flush=True)
    print(f"Huber beta: {args.huber_beta_cp:g} cp", flush=True)
    print_metrics(
        "INITIAL RAW-RUNTIME CALIBRATION",
        evaluate_linear(
            model,
            validation,
            device,
            k_cp,
            args.cp_clip,
            args.validation_batch_size,
        ),
    )

    args.run_dir.mkdir(parents=True, exist_ok=True)
    objective_config = {
        "objective": "linear_huber_cp",
        "init_checkpoint": str(args.init),
        "cp_clip": args.cp_clip,
        "huber_beta_cp": args.huber_beta_cp,
        "k_cp": k_cp,
        "global_start": global_start,
        "global_end": global_end,
        "start_lr": args.start_lr,
        "end_lr_factor": args.end_lr_factor,
        "batch_size": batch_size,
        "seed": args.seed,
    }
    (args.run_dir / "linear_objective_config.json").write_text(
        json.dumps(objective_config, indent=2),
        encoding="utf-8",
    )

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    batch_stm = np.full(
        (batch_size, train_v1.MAX_ACTIVE_FEATURES),
        train_v1.PADDING_INDEX,
        dtype=np.int16,
    )
    batch_opp = np.full(
        (batch_size, train_v1.MAX_ACTIVE_FEATURES),
        train_v1.PADDING_INDEX,
        dtype=np.int16,
    )
    batch_counts = np.empty(batch_size, dtype=np.int16)
    batch_targets = np.empty(batch_size, dtype=np.float32)
    batch_weights = np.empty(batch_size, dtype=np.float32)

    session_seen = 0
    batch_index = 0
    epoch = 0
    next_log = log_every
    next_checkpoint = checkpoint_every
    total_loss = 0.0
    total_loss_positions = 0
    start_time = time.time()

    def current_lr() -> float:
        progress = min(session_seen / max(1, additional), 1.0)
        return args.start_lr * (
            1.0 - progress * (1.0 - args.end_lr_factor)
        )

    while session_seen < additional:
        epoch += 1
        epoch_shards = list(shards)
        rng.shuffle(epoch_shards)
        print(f"Starting epoch {epoch}, session={session_seen:,}", flush=True)

        for shard_path in epoch_shards:
            for line in train_v1.read_shard(
                shard_path,
                args.shuffle_buffer,
                rng,
            ):
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 6:
                    continue

                fen, cp_text, mate_text, depth_text, _knodes, piece_count_text = parts

                try:
                    depth = int(depth_text)
                    metadata_piece_count = int(piece_count_text)
                except ValueError:
                    continue

                if depth < min_depth:
                    continue
                if metadata_piece_count > train_v1.MAX_STANDARD_PIECES:
                    continue

                encoded = train_v1.encode_fen(fen)
                if encoded is None:
                    continue

                stm_indices, opp_indices, piece_count, _material, stm = encoded
                if piece_count != metadata_piece_count:
                    continue

                try:
                    cp_white = None if cp_text == "" else float(cp_text)
                    mate_white = None if mate_text == "" else float(mate_text)
                except ValueError:
                    continue

                result = linear_target(
                    stm,
                    cp_white,
                    mate_white,
                    k_cp,
                    args.cp_clip,
                )
                if result is None:
                    continue

                target, is_mate, cp_stm = result

                batch_stm[batch_index].fill(train_v1.PADDING_INDEX)
                batch_opp[batch_index].fill(train_v1.PADDING_INDEX)
                batch_stm[batch_index, :len(stm_indices)] = stm_indices
                batch_opp[batch_index, :len(opp_indices)] = opp_indices
                batch_counts[batch_index] = piece_count
                batch_targets[batch_index] = target
                batch_weights[batch_index] = train_v1.sample_weight(
                    data_profile,
                    piece_count,
                    cp_stm,
                    is_mate,
                )
                batch_index += 1

                if batch_index < batch_size and session_seen + batch_index < additional:
                    continue

                effective = batch_index
                stm_t = torch.from_numpy(batch_stm[:effective]).to(
                    device=device,
                    dtype=torch.long,
                )
                opp_t = torch.from_numpy(batch_opp[:effective]).to(
                    device=device,
                    dtype=torch.long,
                )
                cnt_t = torch.from_numpy(batch_counts[:effective]).to(
                    device=device,
                    dtype=torch.long,
                )
                target_t = torch.from_numpy(batch_targets[:effective]).to(
                    device=device,
                    dtype=torch.float32,
                )
                weight_t = torch.from_numpy(batch_weights[:effective]).to(
                    device=device,
                    dtype=torch.float32,
                )
                weight_t = weight_t / weight_t.mean()

                optimizer.zero_grad(set_to_none=True)
                predictions = model(stm_t, opp_t, cnt_t)
                beta = args.huber_beta_cp / k_cp
                per_item = F.smooth_l1_loss(
                    predictions,
                    target_t,
                    beta=beta,
                    reduction="none",
                )
                loss = (per_item * weight_t).mean()
                loss.backward()
                optimizer.step()

                session_seen += effective
                total_loss += float(loss.item()) * effective
                total_loss_positions += effective
                batch_index = 0

                lr = current_lr()
                train_v1.set_optimizer_lr(optimizer, lr)

                if session_seen >= next_log:
                    elapsed = time.time() - start_time
                    rate = session_seen / max(elapsed, 1e-9)
                    avg_loss = total_loss / max(total_loss_positions, 1)
                    print(
                        f"session={session_seen:,} "
                        f"global={global_start + session_seen:,} "
                        f"loss={avg_loss:.5f} lr={lr:.2e} "
                        f"rate={rate:,.0f}/s elapsed={elapsed/60:.1f}m",
                        flush=True,
                    )
                    while next_log <= session_seen:
                        next_log += log_every

                if session_seen >= next_checkpoint or session_seen >= additional:
                    metrics = evaluate_linear(
                        model,
                        validation,
                        device,
                        k_cp,
                        args.cp_clip,
                        args.validation_batch_size,
                    )
                    print_metrics(
                        f"VALIDATION @ {format_count(session_seen)} LINEAR POSITIONS",
                        metrics,
                    )

                    name = f"checkpoint_linear_{format_count(session_seen)}.pt"
                    save_checkpoint(
                        args.run_dir / name,
                        model,
                        optimizer,
                        base_config,
                        objective_config,
                        global_start + session_seen,
                        session_seen,
                        metrics,
                    )
                    print(f"Saved {args.run_dir / name}", flush=True)

                    while next_checkpoint <= session_seen:
                        next_checkpoint += checkpoint_every

                if session_seen >= additional:
                    break

            if session_seen >= additional:
                break

    print("LINEAR RECALIBRATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()

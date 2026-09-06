"""Verify linear-Huber PyTorch output against exported NPZ runtime arithmetic."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from training import train_v1
from training.train_v1_linear_recalibrate import LinearV1Evaluator


def npz_eval(data, stm_indices, opp_indices, piece_count):
    embedding = data["embedding"]
    hidden_bias = data["hidden_bias"]
    output_weight = data["output_weight"]
    output_bias = data["output_bias"]
    k_cp = float(np.asarray(data["k_cp"]).item())

    stm = embedding[np.asarray(stm_indices, dtype=np.int64)].sum(axis=0)
    opp = embedding[np.asarray(opp_indices, dtype=np.int64)].sum(axis=0)

    stm = np.clip(stm + hidden_bias, 0.0, 1.0) ** 2
    opp = np.clip(opp + hidden_bias, 0.0, 1.0) ** 2

    buckets = int(np.asarray(data["buckets"]).item())
    if buckets == 1:
        bucket = 0
    else:
        bucket = min(max((piece_count - 2) // 4, 0), buckets - 1)

    hidden = len(stm)
    raw = float(output_bias[bucket])
    raw += float(np.dot(output_weight[bucket, :hidden], stm))
    raw += float(np.dot(output_weight[bucket, hidden:], opp))

    return int(round(max(-4000.0, min(4000.0, raw * k_cp))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("npz", type=Path)
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("training/data/samples/lichess_validation_250k.csv"),
    )
    parser.add_argument("--positions", type=int, default=250)
    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint.get("config", {})
    k_cp = float(config.get("k_cp", 400.0))

    model = LinearV1Evaluator(
        hidden_size=int(config.get("hidden", 64)),
        output_buckets=int(config.get("buckets", 8)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    data = np.load(args.npz, allow_pickle=False)
    errors = []

    with args.validation.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            encoded = train_v1.encode_fen(row["fen"])
            if encoded is None:
                continue

            stm, opp, count, _material, _side = encoded
            stm_t = torch.from_numpy(
                np.expand_dims(train_v1.padded_feature_row(stm), 0)
            ).long()
            opp_t = torch.from_numpy(
                np.expand_dims(train_v1.padded_feature_row(opp), 0)
            ).long()
            cnt_t = torch.tensor([count]).long()

            with torch.no_grad():
                expected = int(
                    round(float(model(stm_t, opp_t, cnt_t).item() * k_cp))
                )
            expected = max(-4000, min(4000, expected))

            actual = npz_eval(data, stm, opp, count)
            errors.append(abs(expected - actual))

            if len(errors) >= args.positions:
                break

    if not errors:
        raise SystemExit("no validation positions checked")

    print(
        f"linear export equivalence: n={len(errors)} "
        f"max_error={max(errors)}cp "
        f"mean_error={np.mean(errors):.3f}cp"
    )

    if max(errors) > 1:
        raise SystemExit(
            "FAIL: exported runtime differs from PyTorch raw model"
        )

    print("PASS")


if __name__ == "__main__":
    main()

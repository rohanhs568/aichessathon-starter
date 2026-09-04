"""Export a trained V1 evaluator checkpoint to compact NumPy inference weights.

Usage:
    python training/export_v1_weights.py \
        training/models/v1_natural_600m/final.pt \
        weights/v1.npz

The exported file contains only inference tensors and small metadata. It does
not contain optimizer state and does not require PyTorch at tournament runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


EXPECTED_INPUT_FEATURES = 772


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if "state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint does not contain state_dict")

    state = checkpoint["state_dict"]
    config = checkpoint.get("config", {})

    required = (
        "embedding.weight",
        "hidden_bias",
        "output.weight",
        "output.bias",
    )
    missing = [name for name in required if name not in state]
    if missing:
        raise RuntimeError(f"Checkpoint is missing tensors: {missing}")

    embedding_full = state["embedding.weight"].detach().cpu().numpy()
    hidden_bias = state["hidden_bias"].detach().cpu().numpy()
    output_weight = state["output.weight"].detach().cpu().numpy()
    output_bias = state["output.bias"].detach().cpu().numpy()

    input_features = int(config.get("input_features", EXPECTED_INPUT_FEATURES))
    hidden = int(config.get("hidden", hidden_bias.shape[0]))
    buckets = int(config.get("buckets", output_bias.shape[0]))
    k_cp = float(config.get("k_cp", 400.0))

    if input_features != EXPECTED_INPUT_FEATURES:
        raise RuntimeError(
            f"Expected {EXPECTED_INPUT_FEATURES} input features, got {input_features}"
        )

    if embedding_full.ndim != 2:
        raise RuntimeError("embedding.weight must be a matrix")
    if embedding_full.shape[0] < input_features:
        raise RuntimeError("embedding.weight has too few rows")
    if embedding_full.shape[1] != hidden:
        raise RuntimeError("Embedding hidden dimension does not match config")
    if hidden_bias.shape != (hidden,):
        raise RuntimeError("hidden_bias shape does not match config")
    if output_weight.shape != (buckets, 2 * hidden):
        raise RuntimeError("output.weight shape does not match config")
    if output_bias.shape != (buckets,):
        raise RuntimeError("output.bias shape does not match config")

    # The final row of the training embedding is only a padding row. Runtime
    # inference uses active feature IDs directly, so it is omitted here.
    embedding = np.ascontiguousarray(
        embedding_full[:input_features],
        dtype=np.float32,
    )
    hidden_bias = np.ascontiguousarray(hidden_bias, dtype=np.float32)
    output_weight = np.ascontiguousarray(output_weight, dtype=np.float32)
    output_bias = np.ascontiguousarray(output_bias, dtype=np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # np.savez is intentionally uncompressed. The file is already tiny and
    # uncompressed arrays load faster during the tournament's init phase.
    np.savez(
        args.output,
        embedding=embedding,
        hidden_bias=hidden_bias,
        output_weight=output_weight,
        output_bias=output_bias,
        input_features=np.asarray(input_features, dtype=np.int32),
        hidden=np.asarray(hidden, dtype=np.int32),
        buckets=np.asarray(buckets, dtype=np.int32),
        k_cp=np.asarray(k_cp, dtype=np.float32),
        format_version=np.asarray(1, dtype=np.int32),
    )

    size = args.output.stat().st_size

    print(f"Exported: {args.output}")
    print(f"Size: {size:,} bytes ({size / 1024:.1f} KiB)")
    print(f"Input features: {input_features}")
    print(f"Hidden: {hidden}")
    print(f"Buckets: {buckets}")
    print(f"K cp: {k_cp:g}")
    print(f"Training positions: {checkpoint.get('positions_seen', 'unknown')}")


if __name__ == "__main__":
    main()

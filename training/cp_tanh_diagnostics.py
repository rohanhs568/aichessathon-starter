"""Audit the V1 tanh target and centipawn measurement logic.

This answers two separate questions:

1) Is runtime raw_output * K the correct inverse of training target
   tanh(cp / K)? Yes, if the model output is tanh(raw), because tanh is
   one-to-one and the optimum satisfies raw = cp / K.

2) Does the current validation metric recover the SAME CP value the runtime
   uses? Not always. train_v1.py clips bounded predictions to +/-0.999 before
   atanh, so |raw| > atanh(0.999) is truncated in reported CP metrics. This
   script quantifies that distortion on a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

from training.train_v1 import V1Evaluator, load_validation


def parse_grid(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def raw_forward(model, stm, opp, counts):
    stm_hidden = model.embedding(stm).sum(dim=1)
    opp_hidden = model.embedding(opp).sum(dim=1)
    stm_hidden = model.screlu(stm_hidden + model.hidden_bias)
    opp_hidden = model.screlu(opp_hidden + model.hidden_bias)
    combined = torch.cat((stm_hidden, opp_hidden), dim=1)
    raw_outputs = model.output(combined)

    if model.output_buckets == 1:
        return raw_outputs[:, 0]

    bucket = torch.div(counts - 2, 4, rounding_mode="floor")
    bucket = torch.clamp(bucket, 0, model.output_buckets - 1)
    return raw_outputs.gather(1, bucket.unsqueeze(1)).squeeze(1)


def safe_mae(pred, target, mask):
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - target[mask])))


def algebra_rows(k_values):
    cps = [0, 50, 100, 200, 400, 600, 800, 1200, 1600, 2000, 3000, 4000]
    rows = []
    for k in k_values:
        for cp in cps:
            raw = cp / k
            bounded = math.tanh(raw)
            recovered = k * math.atanh(bounded)
            target_sensitivity = (1.0 - bounded * bounded) / k
            rows.append({
                "k": k,
                "cp": cp,
                "raw_cp_over_k": raw,
                "target_tanh": bounded,
                "recovered_cp_no_clip": recovered,
                "abs_recovery_error": abs(recovered - cp),
                "dtarget_dcp": target_sensitivity,
            })
    return rows


def scale_summary(k_values, teacher_cp=None):
    rows = []
    for k in k_values:
        clip_cp = k * math.atanh(0.999)
        row = {
            "k": k,
            "legacy_0.999_clip_cp": clip_cp,
            "target_100": math.tanh(100 / k),
            "target_400": math.tanh(400 / k),
            "target_800": math.tanh(800 / k),
            "target_1200": math.tanh(1200 / k),
            "target_2000": math.tanh(2000 / k),
            "gap_800_to_1200": math.tanh(1200 / k) - math.tanh(800 / k),
            "gap_1200_to_2000": math.tanh(2000 / k) - math.tanh(1200 / k),
        }
        if teacher_cp is not None and len(teacher_cp):
            targets = np.tanh(teacher_cp / k)
            row.update({
                "validation_abs_target_ge_0.90_pct": 100 * float(np.mean(np.abs(targets) >= 0.90)),
                "validation_abs_target_ge_0.95_pct": 100 * float(np.mean(np.abs(targets) >= 0.95)),
                "validation_abs_target_ge_0.99_pct": 100 * float(np.mean(np.abs(targets) >= 0.99)),
                "validation_mean_abs_dtarget_dcp": float(np.mean((1.0 - targets * targets) / k)),
            })
        rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def audit_checkpoint(checkpoint_path, validation_path, device_name, batch_size):
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    k = float(config.get("k_cp", 400.0))
    hidden = int(config.get("hidden", 64))
    buckets = int(config.get("buckets", 8))

    model = V1Evaluator(hidden_size=hidden, output_buckets=buckets).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    validation = load_validation(validation_path, k)
    stm_X, opp_X, counts, targets, cp_stm, material, mate_flags = validation

    raw_parts = []
    bounded_parts = []

    with torch.no_grad():
        for start in range(0, len(targets), batch_size):
            end = min(start + batch_size, len(targets))
            stm = torch.from_numpy(stm_X[start:end]).to(device=device, dtype=torch.long)
            opp = torch.from_numpy(opp_X[start:end]).to(device=device, dtype=torch.long)
            cnt = torch.from_numpy(counts[start:end]).to(device=device, dtype=torch.long)
            raw = raw_forward(model, stm, opp, cnt)
            raw_parts.append(raw.cpu().numpy())
            bounded_parts.append(torch.tanh(raw).cpu().numpy())

    raw = np.concatenate(raw_parts).astype(np.float64)
    bounded = np.concatenate(bounded_parts).astype(np.float64)

    cp_mask = ~mate_flags
    teacher = cp_stm[cp_mask].astype(np.float64)
    raw_cp = raw[cp_mask] * k

    clipped = np.clip(bounded[cp_mask], -0.999, 0.999)
    legacy_cp = k * np.arctanh(clipped)

    balanced = np.abs(teacher) <= 400
    medium = np.abs(teacher) <= 800
    decisive = np.abs(teacher) >= 50

    clip_raw = math.atanh(0.999)
    affected = np.abs(raw[cp_mask]) > clip_raw

    result = {
        "checkpoint": str(checkpoint_path),
        "positions_seen": int(checkpoint.get("positions_seen", -1)),
        "k_cp": k,
        "hidden": hidden,
        "buckets": buckets,
        "cp_positions": int(np.sum(cp_mask)),
        "legacy_clip_cp": k * clip_raw,
        "predictions_affected_by_legacy_clip": int(np.sum(affected)),
        "predictions_affected_pct": 100 * float(np.mean(affected)),
        "exact_raw_cp_mae": safe_mae(raw_cp, teacher, np.ones_like(teacher, dtype=bool)),
        "legacy_clipped_cp_mae": safe_mae(legacy_cp, teacher, np.ones_like(teacher, dtype=bool)),
        "exact_balanced_cp_mae": safe_mae(raw_cp, teacher, balanced),
        "legacy_balanced_cp_mae": safe_mae(legacy_cp, teacher, balanced),
        "exact_medium_cp_mae": safe_mae(raw_cp, teacher, medium),
        "legacy_medium_cp_mae": safe_mae(legacy_cp, teacher, medium),
        "exact_sign_accuracy": float(np.mean(np.sign(raw_cp[decisive]) == np.sign(teacher[decisive]))),
        "legacy_sign_accuracy": float(np.mean(np.sign(legacy_cp[decisive]) == np.sign(teacher[decisive]))),
        "max_abs_raw_cp_prediction": float(np.max(np.abs(raw_cp))),
        "mean_abs_exact_minus_legacy_cp": float(np.mean(np.abs(raw_cp - legacy_cp))),
        "max_abs_exact_minus_legacy_cp": float(np.max(np.abs(raw_cp - legacy_cp))),
    }

    return result, teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--validation", type=Path, default=Path("training/data/samples/lichess_validation_250k.csv"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--k-grid", default="400,600,800,1000")
    parser.add_argument("--output-dir", type=Path, default=Path("cp_tanh_diagnostics"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    k_values = parse_grid(args.k_grid)

    checkpoint_result = None
    teacher = None
    if args.checkpoint is not None:
        checkpoint_result, teacher = audit_checkpoint(
            args.checkpoint,
            args.validation,
            args.device,
            args.batch_size,
        )

    algebra = algebra_rows(k_values)
    scales = scale_summary(k_values, teacher)
    write_csv(args.output_dir / "tanh_algebra.csv", algebra)
    write_csv(args.output_dir / "tanh_scale_summary.csv", scales)

    lines = []
    lines.append("CP / TANH DIAGNOSTIC")
    lines.append("=" * 72)
    lines.append("Identity being checked: model=tanh(raw), target=tanh(cp/K) => raw*K = cp at the optimum.")
    lines.append("")

    for k in k_values:
        row = next(r for r in scales if r["k"] == k)
        lines.append(
            f"K={k:g}: clip@0.999={row['legacy_0.999_clip_cp']:.1f}cp, "
            f"tanh(800/K)={row['target_800']:.5f}, "
            f"tanh(1200/K)={row['target_1200']:.5f}, "
            f"gap800->1200={row['gap_800_to_1200']:.5f}"
        )
        if teacher is not None:
            lines.append(
                f"       validation saturated >=.95: {row['validation_abs_target_ge_0.95_pct']:.2f}%  "
                f">=.99: {row['validation_abs_target_ge_0.99_pct']:.2f}%"
            )

    if checkpoint_result is not None:
        lines.append("")
        lines.append("CHECKPOINT AUDIT")
        for key, value in checkpoint_result.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.6f}")
            else:
                lines.append(f"{key}: {value}")
        lines.append("")
        lines.append("Interpretation:")
        lines.append("- exact_raw_cp_* is the score the runtime actually uses before its +/-4000 clamp.")
        lines.append("- legacy_clipped_cp_* reproduces train_v1.py's clip(tanh(raw), +/-0.999) -> atanh path.")
        lines.append("- Any difference is measurement distortion, not a change to the trained model.")

    report = args.output_dir / "cp_tanh_diagnostic.txt"
    report.write_text("\n".join(lines) + "\n")
    print(report.read_text())
    print(f"Wrote {args.output_dir / 'tanh_algebra.csv'}")
    print(f"Wrote {args.output_dir / 'tanh_scale_summary.csv'}")


if __name__ == "__main__":
    main()

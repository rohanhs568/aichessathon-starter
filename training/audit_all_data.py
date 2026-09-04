"""Audit the local Stockfish/Lichess and LC0 training corpora.

This is a read-only integrity/inventory pass. It does not modify either dataset.

Checks:
  * Stockfish shard count and compressed size.
  * Optional zstd integrity test over every Stockfish shard.
  * Random Stockfish row parsing/feature/label checks using train_v1.py.
  * Permanent validation-set row count and V1 encoding sanity.
  * Exact LC0 tar/member count.
  * Exact LC0 record count from each gzip member's ISIZE footer.
  * LC0 V6 record-size divisibility.
  * Sampled LC0 record version/input format/range/board-plane sanity.
  * Free disk space for the training/output locations.

The LC0 count does NOT decompress the full 182 GB corpus. Tar headers and each
small gzip member's four-byte ISIZE footer are enough to count V6 records.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import shutil
import struct
import subprocess
import tarfile
import time
from pathlib import Path

from training import train_v1


LC0_V6_RECORD_SIZE = 8356
LC0_PLANES_OFFSET = 7440
LC0_CASTLING_OFFSET = 8272
LC0_STM_OFFSET = 8276
LC0_ROOT_Q_OFFSET = 8280
LC0_BEST_Q_OFFSET = 8284
LC0_ROOT_D_OFFSET = 8288
LC0_BEST_D_OFFSET = 8292
LC0_RESULT_Q_OFFSET = 8308
LC0_RESULT_D_OFFSET = 8312
LC0_VISITS_OFFSET = 8340


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(value)
    for unit in units:
        if x < 1024.0 or unit == units[-1]:
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} TiB"


def disk_report(path: Path) -> dict:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_human": human_bytes(usage.free),
    }


def parse_stockfish_line(line: str) -> tuple[bool, str]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 6:
        return False, f"expected 6 TSV fields, got {len(parts)}"

    fen, cp_text, mate_text, depth_text, knodes_text, piece_count_text = parts

    try:
        depth = int(depth_text)
        int(knodes_text)
        metadata_piece_count = int(piece_count_text)
    except ValueError as exc:
        return False, f"bad integer metadata: {exc}"

    encoded = train_v1.encode_fen(fen)
    if encoded is None:
        return False, "V1 encoder rejected FEN"

    _stm_ids, _opp_ids, piece_count, _material, stm = encoded
    if piece_count != metadata_piece_count:
        return False, f"piece_count mismatch {piece_count} != {metadata_piece_count}"

    try:
        cp_white = None if cp_text == "" else float(cp_text)
        mate_white = None if mate_text == "" else float(mate_text)
    except ValueError as exc:
        return False, f"bad label: {exc}"

    if cp_white is None and mate_white is None:
        return False, "neither cp nor mate label present"

    target = train_v1.target_from_label(
        stm,
        cp_white,
        mate_white,
        400.0,
    )
    if target is None:
        return False, "target conversion failed"

    if depth < 0:
        return False, f"negative depth {depth}"

    return True, ""


def sample_stockfish_shards(
    shard_paths: list[Path],
    samples_per_shard: int,
    shard_sample_count: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    selected = (
        shard_paths
        if len(shard_paths) <= shard_sample_count
        else rng.sample(shard_paths, shard_sample_count)
    )

    checked = 0
    bad = []

    for path in selected:
        process = subprocess.Popen(
            ["zstd", "-dc", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1024 * 1024,
        )

        try:
            for _ in range(samples_per_shard):
                line = process.stdout.readline()
                if not line:
                    break
                checked += 1
                ok, reason = parse_stockfish_line(line)
                if not ok:
                    bad.append(
                        {
                            "shard": str(path),
                            "reason": reason,
                            "line_prefix": line[:200],
                        }
                    )
                    if len(bad) >= 20:
                        break
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        if len(bad) >= 20:
            break

    return {
        "sampled_shards": len(selected),
        "sampled_rows": checked,
        "bad_rows": bad,
    }


def zstd_integrity(shard_paths: list[Path]) -> dict:
    failures = []
    start = time.time()

    for index, path in enumerate(shard_paths, start=1):
        result = subprocess.run(
            ["zstd", "-t", "-q", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            failures.append(
                {
                    "path": str(path),
                    "stderr": result.stderr[-1000:],
                }
            )

        if index % 25 == 0 or index == len(shard_paths):
            print(
                f"  zstd integrity: {index}/{len(shard_paths)}",
                flush=True,
            )

    return {
        "files_tested": len(shard_paths),
        "failures": failures,
        "elapsed_seconds": time.time() - start,
    }


def audit_validation(path: Path) -> dict:
    rows = 0
    encoded_ok = 0
    rejected = 0
    bad_labels = 0

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        required = {"fen", "cp_white", "mate_white"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                f"validation CSV missing columns: {sorted(missing)}"
            )

        for row in reader:
            rows += 1
            encoded = train_v1.encode_fen(row["fen"])
            if encoded is None:
                rejected += 1
                continue

            encoded_ok += 1

            cp_text = row["cp_white"]
            mate_text = row["mate_white"]
            if cp_text == "" and mate_text == "":
                bad_labels += 1

    return {
        "rows": rows,
        "encoded_ok": encoded_ok,
        "encoder_rejected": rejected,
        "missing_label_rows": bad_labels,
    }


def read_gzip_isize(
    handle,
    offset_data: int,
    compressed_size: int,
) -> int:
    if compressed_size < 4:
        raise ValueError("gzip member shorter than four bytes")

    handle.seek(offset_data + compressed_size - 4)
    footer = handle.read(4)

    if len(footer) != 4:
        raise ValueError("could not read gzip ISIZE footer")

    return struct.unpack("<I", footer)[0]


def first_set_bit(mask: int) -> int | None:
    if mask == 0:
        return None
    return (mask & -mask).bit_length() - 1


def validate_lc0_record(record: bytes) -> tuple[bool, str]:
    if len(record) != LC0_V6_RECORD_SIZE:
        return False, f"record length {len(record)}"

    version, input_format = struct.unpack_from("<II", record, 0)

    if version != 6:
        return False, f"version={version}"

    if input_format != 1:
        return False, f"input_format={input_format}"

    planes = struct.unpack_from("<104Q", record, LC0_PLANES_OFFSET)

    current = planes[:12]
    occupancy = 0

    for index, mask in enumerate(current):
        if occupancy & mask:
            return False, f"overlapping current-position piece planes at {index}"
        occupancy |= mask

    our_king = current[5]
    their_king = current[11]

    if our_king.bit_count() != 1:
        return False, f"our king count={our_king.bit_count()}"

    if their_king.bit_count() != 1:
        return False, f"their king count={their_king.bit_count()}"

    piece_count = occupancy.bit_count()
    if not 2 <= piece_count <= 32:
        return False, f"piece_count={piece_count}"

    castling = record[
        LC0_CASTLING_OFFSET : LC0_CASTLING_OFFSET + 4
    ]
    if any(value not in (0, 1) for value in castling):
        return False, f"bad castling bytes={list(castling)}"

    stm = record[LC0_STM_OFFSET]
    if stm not in (0, 1):
        return False, f"side_to_move={stm}"

    (
        root_q,
        best_q,
        root_d,
        best_d,
    ) = struct.unpack_from(
        "<ffff",
        record,
        LC0_ROOT_Q_OFFSET,
    )

    result_q = struct.unpack_from(
        "<f",
        record,
        LC0_RESULT_Q_OFFSET,
    )[0]
    result_d = struct.unpack_from(
        "<f",
        record,
        LC0_RESULT_D_OFFSET,
    )[0]
    visits = struct.unpack_from(
        "<I",
        record,
        LC0_VISITS_OFFSET,
    )[0]

    for name, value in (
        ("root_q", root_q),
        ("best_q", best_q),
        ("result_q", result_q),
    ):
        if not math.isfinite(value) or not -1.0001 <= value <= 1.0001:
            return False, f"{name}={value}"

    for name, value in (
        ("root_d", root_d),
        ("best_d", best_d),
        ("result_d", result_d),
    ):
        if not math.isfinite(value) or not -0.0001 <= value <= 1.0001:
            return False, f"{name}={value}"

    if visits == 0:
        return False, "visits=0"

    return True, ""


def audit_lc0(
    root: Path,
    sample_members_per_tar: int,
) -> tuple[dict, list[dict]]:
    tar_paths = sorted(root.glob("*.tar"))

    if not tar_paths:
        raise RuntimeError(f"No .tar archives found in {root}")

    total_members = 0
    total_compressed = 0
    total_uncompressed = 0
    total_records = 0
    bad_isize = []
    bad_samples = []
    input_formats = {}
    versions = {}
    per_tar = []

    start = time.time()

    for tar_index, tar_path in enumerate(tar_paths, start=1):
        tar_members = 0
        tar_compressed = 0
        tar_uncompressed = 0
        tar_records = 0
        sampled = 0

        with (
            tarfile.open(tar_path, mode="r:") as archive,
            tar_path.open("rb") as footer_handle,
        ):
            for member in archive:
                if not member.isfile() or not member.name.endswith(".gz"):
                    continue

                tar_members += 1
                total_members += 1
                tar_compressed += member.size
                total_compressed += member.size

                try:
                    isize = read_gzip_isize(
                        footer_handle,
                        member.offset_data,
                        member.size,
                    )
                except Exception as exc:
                    bad_isize.append(
                        {
                            "tar": str(tar_path),
                            "member": member.name,
                            "reason": repr(exc),
                        }
                    )
                    continue

                tar_uncompressed += isize
                total_uncompressed += isize

                if isize % LC0_V6_RECORD_SIZE != 0:
                    bad_isize.append(
                        {
                            "tar": str(tar_path),
                            "member": member.name,
                            "isize": isize,
                            "reason": "not divisible by 8356",
                        }
                    )
                    continue

                records = isize // LC0_V6_RECORD_SIZE
                tar_records += records
                total_records += records

                if sampled < sample_members_per_tar:
                    fileobj = archive.extractfile(member)
                    if fileobj is None:
                        bad_samples.append(
                            {
                                "tar": str(tar_path),
                                "member": member.name,
                                "reason": "extractfile returned None",
                            }
                        )
                        continue

                    compressed = fileobj.read()

                    try:
                        payload = gzip.decompress(compressed)
                    except Exception as exc:
                        bad_samples.append(
                            {
                                "tar": str(tar_path),
                                "member": member.name,
                                "reason": f"gzip: {exc!r}",
                            }
                        )
                        continue

                    if len(payload) != isize:
                        bad_samples.append(
                            {
                                "tar": str(tar_path),
                                "member": member.name,
                                "reason": (
                                    f"decompressed {len(payload)} != ISIZE {isize}"
                                ),
                            }
                        )
                        continue

                    for offset in range(
                        0,
                        len(payload),
                        LC0_V6_RECORD_SIZE,
                    ):
                        record = payload[
                            offset : offset + LC0_V6_RECORD_SIZE
                        ]
                        if len(record) != LC0_V6_RECORD_SIZE:
                            bad_samples.append(
                                {
                                    "tar": str(tar_path),
                                    "member": member.name,
                                    "reason": "partial record",
                                }
                            )
                            break

                        version, input_format = struct.unpack_from(
                            "<II",
                            record,
                            0,
                        )
                        versions[version] = versions.get(version, 0) + 1
                        input_formats[input_format] = (
                            input_formats.get(input_format, 0) + 1
                        )

                        ok, reason = validate_lc0_record(record)
                        if not ok:
                            bad_samples.append(
                                {
                                    "tar": str(tar_path),
                                    "member": member.name,
                                    "reason": reason,
                                }
                            )
                            if len(bad_samples) >= 50:
                                break

                    sampled += 1

                if len(bad_samples) >= 50:
                    break

        per_tar.append(
            {
                "tar": str(tar_path),
                "gzip_members": tar_members,
                "compressed_bytes": tar_compressed,
                "uncompressed_bytes": tar_uncompressed,
                "records": tar_records,
            }
        )

        print(
            f"  LC0 inventory {tar_index}/{len(tar_paths)}: "
            f"{tar_records:,} records in {tar_members:,} chunks",
            flush=True,
        )

        if len(bad_samples) >= 50:
            break

    summary = {
        "tar_files": len(tar_paths),
        "gzip_members": total_members,
        "compressed_bytes": total_compressed,
        "compressed_human": human_bytes(total_compressed),
        "uncompressed_bytes": total_uncompressed,
        "uncompressed_human": human_bytes(total_uncompressed),
        "records": total_records,
        "bad_isize_members": bad_isize,
        "bad_sample_records": bad_samples,
        "sampled_versions": versions,
        "sampled_input_formats": input_formats,
        "elapsed_seconds": time.time() - start,
    }

    return summary, per_tar


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stockfish-shards",
        type=Path,
        default=Path("/mnt/d/ChessData/lichess_train_shards"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path(
            "training/data/samples/lichess_validation_250k.csv"
        ),
    )
    parser.add_argument(
        "--lc0-root",
        type=Path,
        default=Path("/mnt/d/ChessData/lc0/test79_random"),
    )
    parser.add_argument(
        "--stockfish-sample-shards",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--stockfish-samples-per-shard",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--lc0-sample-members-per-tar",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--full-zstd-test",
        action="store_true",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("training/data/data_audit.json"),
    )
    parser.add_argument(
        "--lc0-csv",
        type=Path,
        default=Path("training/data/lc0_inventory.csv"),
    )
    parser.add_argument("--seed", type=int, default=20260904)

    args = parser.parse_args()

    shard_paths = sorted(
        args.stockfish_shards.glob("lichess_train_*.tsv.zst")
    )
    if not shard_paths:
        raise SystemExit(
            f"No Stockfish shards found in {args.stockfish_shards}"
        )

    print("Stockfish/Lichess corpus")
    print("-----------------------")
    sf_bytes = sum(path.stat().st_size for path in shard_paths)
    print(f"shards: {len(shard_paths):,}")
    print(f"compressed size: {human_bytes(sf_bytes)}")

    sf_samples = sample_stockfish_shards(
        shard_paths,
        args.stockfish_samples_per_shard,
        args.stockfish_sample_shards,
        args.seed,
    )
    print(
        f"sample parse: {sf_samples['sampled_rows']:,} rows, "
        f"{len(sf_samples['bad_rows'])} bad"
    )

    validation = audit_validation(args.validation)
    print(
        f"validation: {validation['rows']:,} rows, "
        f"{validation['encoded_ok']:,} encoded, "
        f"{validation['encoder_rejected']:,} rejected"
    )

    zstd_result = None
    if args.full_zstd_test:
        print("Running full zstd integrity test...", flush=True)
        zstd_result = zstd_integrity(shard_paths)
        print(
            f"zstd failures: {len(zstd_result['failures'])}",
            flush=True,
        )

    print()
    print("LC0 corpus")
    print("----------")
    lc0_summary, per_tar = audit_lc0(
        args.lc0_root,
        args.lc0_sample_members_per_tar,
    )

    print(f"tar files: {lc0_summary['tar_files']:,}")
    print(f"gzip chunks: {lc0_summary['gzip_members']:,}")
    print(f"exact V6 records: {lc0_summary['records']:,}")
    print(f"compressed payload: {lc0_summary['compressed_human']}")
    print(f"uncompressed payload: {lc0_summary['uncompressed_human']}")
    print(f"sampled versions: {lc0_summary['sampled_versions']}")
    print(
        f"sampled input formats: "
        f"{lc0_summary['sampled_input_formats']}"
    )
    print(
        f"bad ISIZE members: "
        f"{len(lc0_summary['bad_isize_members'])}"
    )
    print(
        f"bad sampled records: "
        f"{len(lc0_summary['bad_sample_records'])}"
    )

    report = {
        "stockfish": {
            "shards": len(shard_paths),
            "compressed_bytes": sf_bytes,
            "compressed_human": human_bytes(sf_bytes),
            "samples": sf_samples,
            "zstd_integrity": zstd_result,
        },
        "validation": validation,
        "lc0": lc0_summary,
        "disk": {
            "repo": disk_report(Path(".")),
            "stockfish": disk_report(args.stockfish_shards),
            "lc0": disk_report(args.lc0_root),
        },
    }

    args.report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.report.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    args.lc0_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with args.lc0_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tar",
                "gzip_members",
                "compressed_bytes",
                "uncompressed_bytes",
                "records",
            ],
        )
        writer.writeheader()
        writer.writerows(per_tar)

    print()
    print("Disk free:")
    for name, item in report["disk"].items():
        print(f"  {name}: {item['free_human']}")

    fatal = (
        len(sf_samples["bad_rows"])
        + len(lc0_summary["bad_isize_members"])
        + len(lc0_summary["bad_sample_records"])
        + validation["missing_label_rows"]
    )
    if zstd_result is not None:
        fatal += len(zstd_result["failures"])

    print()
    print(f"Audit report: {args.report}")
    print(f"LC0 inventory: {args.lc0_csv}")
    print(f"fatal/anomalous findings: {fatal}")

    if fatal:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""Paired search-feature ablation suite for the Chessathon V1 FastQ engine.

Primary metric: Stockfish WDL expectation loss from the ROOT player's point of
view. Secondary metric: centipawn loss, also from the root player's POV.

Why both? Centipawn differences are intuitive around equal positions, but are
not linear in game outcome. A +100 -> -100 swing matters much more than a
+300 -> +500 difference of the same 200 cp. python-chess explicitly recommends
WDL expectation differences for this reason.

The candidate move is evaluated by Stockfish on the ORIGINAL root board using
root_moves=[candidate]. This avoids the common sign bug caused by pushing the
candidate and forgetting that side-to-move has flipped.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import os
import random
import re
import statistics
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine


CORE_VARIANTS = [
    "baseline",
    "no_null",
    "null_safe",
    "null_no_endgame",
    "no_lmr",
    "lmr_safe",
    "no_futility",
    "no_selectivity",
    "conservative",
]

WIDE_EXTRA_VARIANTS = [
    "futility_wide",
    "capture_material_order",
    "no_aspiration",
    "no_pvs",
]

DEPTH_RE = re.compile(r"completed_depth=(\d+)")
SCORE_RE = re.compile(r"depth=(\d+) score=(-?\d+)")
NODES_RE = re.compile(r"total_nodes=(\d+)")


@dataclass
class Case:
    case_id: str
    fen: str
    category: str = "unknown"
    source: str = ""


def load_cases(fens_path: Path | None, cases_csv: Path | None, max_fens: int | None):
    cases: list[Case] = []

    if fens_path is not None:
        with fens_path.open() as f:
            for index, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                # Existing test_fens_60.txt is one FEN per line.
                chess.Board(line)  # validate early
                cases.append(Case(f"fen_{index:04d}", line, "fen_suite", str(fens_path)))

    if cases_csv is not None:
        with cases_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            if "fen" not in (reader.fieldnames or []):
                raise ValueError("cases CSV must contain a 'fen' column")
            for index, row in enumerate(reader, start=1):
                fen = row["fen"].strip()
                if not fen:
                    continue
                chess.Board(fen)
                cases.append(
                    Case(
                        row.get("id") or row.get("case_id") or f"csv_{index:04d}",
                        fen,
                        row.get("category") or "custom",
                        row.get("source") or str(cases_csv),
                    )
                )

    if not cases:
        raise ValueError("no cases loaded; pass --fens and/or --cases-csv")

    # De-duplicate exact FENs but preserve first metadata occurrence.
    unique = []
    seen = set()
    for case in cases:
        if case.fen in seen:
            continue
        seen.add(case.fen)
        unique.append(case)

    if max_fens is not None:
        unique = unique[:max_fens]

    return unique


def load_lab_agent(path: Path, weights: Path | None):
    if weights is not None:
        os.environ["SEARCHLAB_MODEL_PATH"] = str(weights.resolve())

    spec = importlib.util.spec_from_file_location("agent_searchlab_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")

    module = importlib.util.module_from_spec(spec)
    with redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def run_candidate(agent, case: Case, variant: str, time_left_ms: int, mode: str, agent_depth: int):
    agent.set_searchlab_variant(variant)
    agent.reset_searchlab_state(reset_game=True)

    if mode == "timed":
        capture = io.StringIO()
        started = time.perf_counter()
        with redirect_stdout(capture):
            move_uci = agent.get_move(case.fen, time_left_ms)
        elapsed = time.perf_counter() - started

        log = capture.getvalue()
        depth_matches = DEPTH_RE.findall(log)
        score_matches = SCORE_RE.findall(log)
        node_matches = NODES_RE.findall(log)

        completed_depth = int(depth_matches[-1]) if depth_matches else 0
        engine_score_cp = int(score_matches[-1][1]) if score_matches else None
        total_nodes = int(node_matches[-1]) if node_matches else int(getattr(agent, "NODES", 0))

    elif mode == "fixed-depth":
        board = chess.Board(case.fen)
        agent._GAME_COUNTS = agent._build_game_counts(board)
        agent._PATH_COUNTS = {}
        agent.DEADLINE = time.monotonic() + 3600.0
        agent.NODES = 0

        started = time.perf_counter()
        best_move = None
        best_score = None
        log_lines = []
        for depth in range(1, agent_depth + 1):
            before = agent.NODES
            move, score = agent.aspiration_search(board, depth, best_score)
            used = agent.NODES - before
            best_move, best_score = move, score
            log_lines.append(
                f"depth={depth} score={score} nodes={used} total_nodes={agent.NODES}"
            )
        elapsed = time.perf_counter() - started
        if best_move is None:
            raise RuntimeError("fixed-depth search returned no move")
        move_uci = best_move.uci()
        completed_depth = agent_depth
        engine_score_cp = int(best_score)
        total_nodes = int(agent.NODES)
        log = "\n".join(log_lines) + "\n"

    else:
        raise ValueError(f"bad mode {mode!r}")

    board = chess.Board(case.fen)
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        raise RuntimeError(f"{variant} returned illegal move {move_uci} for {case.case_id}")

    stats = agent.get_searchlab_stats()

    return {
        "move": move_uci,
        "mode": mode,
        "elapsed_s": elapsed,
        "completed_depth": completed_depth,
        "engine_score_cp": engine_score_cp,
        "nodes": total_nodes,
        "nps": total_nodes / elapsed if elapsed > 0 else 0.0,
        "agent_log": log,
        **stats,
    }


def configure_stockfish(engine, hash_mb: int):
    options = {}
    if "Threads" in engine.options:
        options["Threads"] = 1
    if "Hash" in engine.options:
        options["Hash"] = hash_mb
    if "UCI_ShowWDL" in engine.options:
        options["UCI_ShowWDL"] = True
    if options:
        engine.configure(options)


def score_record(info: dict, root_color: chess.Color, board_ply: int):
    pov = info["score"].pov(root_color)
    cp_equiv = pov.score(mate_score=100_000)
    mate = pov.mate()

    # Prefer WDL supplied by the running Stockfish build. Fall back to the
    # python-chess Stockfish WDL model when UCI_ShowWDL is unavailable.
    if "wdl" in info:
        wdl = info["wdl"].pov(root_color)
    else:
        wdl = pov.wdl(model="sf", ply=board_ply)

    expectation = float(wdl.expectation())

    pv = [move.uci() for move in info.get("pv", [])]

    return {
        "score_text": str(pov),
        "cp_equiv": int(cp_equiv) if cp_equiv is not None else None,
        "mate": mate,
        "is_mate": bool(pov.is_mate()),
        "expectation": expectation,
        "wdl_wins": int(wdl.wins),
        "wdl_draws": int(wdl.draws),
        "wdl_losses": int(wdl.losses),
        "pv": pv,
        "depth": int(info.get("depth", 0)),
        "seldepth": int(info.get("seldepth", 0)),
        "nodes": int(info.get("nodes", 0)),
    }


def load_cache(path: Path):
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def save_cache(path: Path, cache):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    tmp.replace(path)


def analyse_best(engine, board, sf_depth, cache, cache_path):
    key = f"best|d={sf_depth}|{board.fen()}"
    if key in cache:
        return cache[key]

    info = engine.analyse(board, chess.engine.Limit(depth=sf_depth))
    record = score_record(info, board.turn, board.ply())
    record["best_move"] = record["pv"][0] if record["pv"] else None
    cache[key] = record
    save_cache(cache_path, cache)
    return record


def analyse_forced_move(engine, board, move, sf_depth, cache, cache_path, best_record):
    if best_record.get("best_move") == move.uci():
        return best_record

    key = f"move|d={sf_depth}|{move.uci()}|{board.fen()}"
    if key in cache:
        return cache[key]

    info = engine.analyse(
        board,
        chess.engine.Limit(depth=sf_depth),
        root_moves=[move],
    )
    record = score_record(info, board.turn, board.ply())
    record["best_move"] = move.uci()
    cache[key] = record
    save_cache(cache_path, cache)
    return record


def percentile(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    weight = pos - lo
    return float(ordered[lo] * (1 - weight) + ordered[hi] * weight)


def summarize(rows, variants):
    result = []
    for variant in variants:
        vr = [r for r in rows if r["variant"] == variant and not r.get("error")]
        if not vr:
            continue

        cp_losses = [float(r["cp_loss"]) for r in vr]
        capped = [min(x, 1000.0) for x in cp_losses]
        exp_losses = [float(r["expectation_loss"]) for r in vr]

        result.append({
            "variant": variant,
            "n": len(vr),
            "exact_best_pct": 100 * sum(r["move"] == r["sf_best_move"] for r in vr) / len(vr),
            "within_10cp_pct": 100 * sum(x <= 10 for x in cp_losses) / len(vr),
            "within_30cp_pct": 100 * sum(x <= 30 for x in cp_losses) / len(vr),
            "within_75cp_pct": 100 * sum(x <= 75 for x in cp_losses) / len(vr),
            "within_150cp_pct": 100 * sum(x <= 150 for x in cp_losses) / len(vr),
            "blunder_300cp_pct": 100 * sum(x >= 300 for x in cp_losses) / len(vr),
            "mean_cp_loss_capped1000": statistics.fmean(capped),
            "median_cp_loss": statistics.median(cp_losses),
            "p90_cp_loss": percentile(cp_losses, 0.90),
            "mean_expectation_loss": statistics.fmean(exp_losses),
            "median_expectation_loss": statistics.median(exp_losses),
            "p90_expectation_loss": percentile(exp_losses, 0.90),
            "mate_blunders": sum(bool(r["mate_blunder"]) for r in vr),
            "winning_to_notwinning": sum(bool(r["winning_to_notwinning"]) for r in vr),
            "sign_flips": sum(bool(r["sign_flip"]) for r in vr),
            "avg_depth": statistics.fmean(float(r["completed_depth"]) for r in vr),
            "avg_nodes": statistics.fmean(float(r["nodes"]) for r in vr),
            "avg_nps": statistics.fmean(float(r["nps"]) for r in vr),
            "avg_elapsed_s": statistics.fmean(float(r["elapsed_s"]) for r in vr),
            "null_tries": sum(int(r.get("null_tries", 0)) for r in vr),
            "null_cutoffs": sum(int(r.get("null_cutoffs", 0)) for r in vr),
            "lmr_reductions": sum(int(r.get("lmr_reductions", 0)) for r in vr),
            "lmr_researches": sum(int(r.get("lmr_researches", 0)) for r in vr),
            "futility_skips": sum(int(r.get("futility_skips", 0)) for r in vr),
            "tt_cutoffs": sum(int(r.get("tt_cutoffs", 0)) for r in vr),
            "aspiration_retries": sum(int(r.get("aspiration_retries", 0)) for r in vr),
        })
    return result


def bootstrap_mean_delta(deltas, samples, seed=12345):
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    if len(deltas) == 1:
        value = float(deltas[0])
        return value, value, value

    rng = random.Random(seed)
    means = []
    n = len(deltas)
    for _ in range(samples):
        means.append(statistics.fmean(deltas[rng.randrange(n)] for _ in range(n)))
    means.sort()
    mean = statistics.fmean(deltas)
    lo = means[int(0.025 * (samples - 1))]
    hi = means[int(0.975 * (samples - 1))]
    return mean, lo, hi


def pairwise_vs_baseline(rows, variants, bootstrap_samples):
    by_key = {(r["variant"], r["case_id"]): r for r in rows if not r.get("error")}
    result = []

    for variant in variants:
        if variant == "baseline":
            continue
        pairs = []
        for (v, case_id), base in by_key.items():
            if v != "baseline":
                continue
            other = by_key.get((variant, case_id))
            if other is not None:
                pairs.append((base, other))

        if not pairs:
            continue

        # Positive delta means the variant is WORSE than baseline.
        exp_deltas = [o["expectation_loss"] - b["expectation_loss"] for b, o in pairs]
        cp_deltas = [o["cp_loss"] - b["cp_loss"] for b, o in pairs]
        mean_exp, lo_exp, hi_exp = bootstrap_mean_delta(exp_deltas, bootstrap_samples)
        mean_cp, lo_cp, hi_cp = bootstrap_mean_delta(cp_deltas, bootstrap_samples, seed=54321)

        result.append({
            "variant": variant,
            "n_pairs": len(pairs),
            "changed_move_pct": 100 * sum(b["move"] != o["move"] for b, o in pairs) / len(pairs),
            "improved_exp_gt_0.01": sum(d < -0.01 for d in exp_deltas),
            "worse_exp_gt_0.01": sum(d > 0.01 for d in exp_deltas),
            "equal_exp_within_0.01": sum(abs(d) <= 0.01 for d in exp_deltas),
            "mean_expectation_loss_delta": mean_exp,
            "exp_delta_ci95_lo": lo_exp,
            "exp_delta_ci95_hi": hi_exp,
            "mean_cp_loss_delta": mean_cp,
            "cp_delta_ci95_lo": lo_cp,
            "cp_delta_ci95_hi": hi_cp,
            "rescued_150cp_blunders": sum(b["cp_loss"] >= 150 and o["cp_loss"] < 75 for b, o in pairs),
            "introduced_150cp_blunders": sum(b["cp_loss"] < 75 and o["cp_loss"] >= 150 for b, o in pairs),
        })

    return result


def write_csv(path: Path, rows):
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, summaries, pairwise, args, cases):
    ranked = sorted(summaries, key=lambda x: (x["mean_expectation_loss"], x["mean_cp_loss_capped1000"]))
    lines = []
    lines.append("SEARCH ABLATION SUMMARY")
    lines.append("=" * 72)
    lines.append(f"cases: {len(cases)}")
    lines.append(f"mode: {args.mode}")
    lines.append(f"agent time_left_ms: {args.time_left_ms}")
    lines.append(f"agent fixed depth: {args.agent_depth}")
    lines.append(f"Stockfish depth: {args.sf_depth}")
    lines.append("")
    lines.append("Primary ranking: lower Stockfish WDL expectation loss is better.")
    lines.append("CP loss is secondary and capped at 1000 cp in the mean.")
    lines.append("")

    for rank, s in enumerate(ranked, start=1):
        lines.append(
            f"{rank:2d}. {s['variant']:<24} "
            f"exp_loss={s['mean_expectation_loss']:.4f} "
            f"cp_loss={s['mean_cp_loss_capped1000']:.1f} "
            f"median={s['median_cp_loss']:.1f} "
            f">=300={s['blunder_300cp_pct']:.1f}% "
            f"depth={s['avg_depth']:.2f} nodes={s['avg_nodes']:.0f}"
        )

    lines.append("")
    lines.append("PAIRED VS BASELINE")
    lines.append("Positive delta = variant worse; negative delta = variant better.")
    for p in pairwise:
        lines.append(
            f"{p['variant']:<24} "
            f"dExp={p['mean_expectation_loss_delta']:+.4f} "
            f"95%CI=[{p['exp_delta_ci95_lo']:+.4f},{p['exp_delta_ci95_hi']:+.4f}] "
            f"dCP={p['mean_cp_loss_delta']:+.1f} "
            f"changed={p['changed_move_pct']:.1f}% "
            f"rescued={p['rescued_150cp_blunders']} "
            f"new_blunders={p['introduced_150cp_blunders']}"
        )

    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-lab", type=Path, default=Path("training/searchlab_agent.py"))
    parser.add_argument("--weights", type=Path, default=Path("weights/v1.npz"))
    parser.add_argument("--fens", type=Path)
    parser.add_argument("--cases-csv", type=Path)
    parser.add_argument("--max-fens", type=int)
    parser.add_argument("--preset", choices=("core", "wide"), default="core")
    parser.add_argument("--variants", help="comma-separated override")
    parser.add_argument("--time-left-ms", type=int, default=120_000)
    parser.add_argument("--mode", choices=("timed", "fixed-depth"), default="timed")
    parser.add_argument("--agent-depth", type=int, default=5, help="used only with --mode fixed-depth")
    parser.add_argument("--stockfish", default="/usr/games/stockfish")
    parser.add_argument("--sf-depth", type=int, default=18)
    parser.add_argument("--sf-hash", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=Path("search_lab_results"))
    args = parser.parse_args()

    cases = load_cases(args.fens, args.cases_csv, args.max_fens)
    if args.variants:
        variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    elif args.preset == "wide":
        variants = CORE_VARIANTS + WIDE_EXTRA_VARIANTS
    else:
        variants = CORE_VARIANTS

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / f"stockfish_depth{args.sf_depth}_cache.json"
    cache = load_cache(cache_path)

    approx_agent_seconds = len(cases) * len(variants) * min(3.5, args.time_left_ms / 1000)
    print(f"cases={len(cases)} variants={len(variants)} mode={args.mode}")
    if args.mode == "timed":
        print(f"rough candidate-search time upper estimate: {approx_agent_seconds/60:.1f} min")
    else:
        print(f"fixed-depth candidate search through depth {args.agent_depth}")
    print("Stockfish reference work is cached by FEN/move, so reruns get cheaper.")

    agent = load_lab_agent(args.agent_lab, args.weights)
    missing = [v for v in variants if v not in agent.SEARCHLAB_PRESETS]
    if missing:
        raise ValueError(f"unknown variants for lab agent: {missing}")

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    configure_stockfish(engine, args.sf_hash)

    rows = []
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    try:
        for case_index, case in enumerate(cases, start=1):
            board = chess.Board(case.fen)
            best = analyse_best(engine, board, args.sf_depth, cache, cache_path)
            print(
                f"[{case_index}/{len(cases)}] {case.case_id} "
                f"SF={best.get('best_move')} {best['score_text']} exp={best['expectation']:.3f}"
            )

            for variant in variants:
                try:
                    cand = run_candidate(agent, case, variant, args.time_left_ms, args.mode, args.agent_depth)
                    move = chess.Move.from_uci(cand["move"])
                    forced = analyse_forced_move(
                        engine, board, move, args.sf_depth, cache, cache_path, best
                    )

                    best_cp = best["cp_equiv"]
                    cand_cp = forced["cp_equiv"]
                    cp_delta_raw = float(best_cp - cand_cp)
                    cp_loss = max(0.0, cp_delta_raw)
                    exp_delta_raw = float(best["expectation"] - forced["expectation"])
                    exp_loss = max(0.0, exp_delta_raw)

                    best_mate = best["mate"]
                    cand_mate = forced["mate"]
                    mate_blunder = bool(cand_mate is not None and cand_mate < 0 and not (best_mate is not None and best_mate < 0))
                    sign_flip = bool(best_cp >= 100 and cand_cp <= -100)
                    winning_to_notwinning = bool(best["expectation"] >= 0.75 and forced["expectation"] < 0.55)

                    row = {
                        "case_id": case.case_id,
                        "category": case.category,
                        "source": case.source,
                        "fen": case.fen,
                        "variant": variant,
                        **{k: v for k, v in cand.items() if k != "agent_log"},
                        "sf_best_move": best.get("best_move"),
                        "sf_best_score": best["score_text"],
                        "sf_candidate_score": forced["score_text"],
                        "sf_best_cp_equiv": best_cp,
                        "sf_candidate_cp_equiv": cand_cp,
                        "cp_delta_raw": cp_delta_raw,
                        "cp_loss": cp_loss,
                        "sf_best_expectation": best["expectation"],
                        "sf_candidate_expectation": forced["expectation"],
                        "expectation_delta_raw": exp_delta_raw,
                        "expectation_loss": exp_loss,
                        "mate_blunder": mate_blunder,
                        "sign_flip": sign_flip,
                        "winning_to_notwinning": winning_to_notwinning,
                        "sf_best_pv": " ".join(best.get("pv", [])[:12]),
                        "sf_candidate_pv": " ".join(forced.get("pv", [])[:12]),
                    }
                    rows.append(row)

                    (logs_dir / f"{case.case_id}__{variant}.txt").write_text(cand["agent_log"])
                    print(
                        f"  {variant:<24} {cand['move']} "
                        f"loss={cp_loss:6.1f}cp exp={exp_loss:.4f} "
                        f"d={cand['completed_depth']} n={cand['nodes']}"
                    )
                except Exception as exc:
                    rows.append({
                        "case_id": case.case_id,
                        "category": case.category,
                        "source": case.source,
                        "fen": case.fen,
                        "variant": variant,
                        "error": repr(exc),
                    })
                    print(f"  {variant:<24} ERROR {exc!r}")

                write_csv(args.output_dir / "search_ablation_raw.csv", rows)
    finally:
        engine.quit()

    summaries = summarize(rows, variants)
    pairwise = pairwise_vs_baseline(rows, variants, args.bootstrap)
    write_csv(args.output_dir / "search_ablation_summary.csv", summaries)
    write_csv(args.output_dir / "search_ablation_pairwise.csv", pairwise)
    write_summary(args.output_dir / "search_ablation_summary.txt", summaries, pairwise, args, cases)

    print()
    print((args.output_dir / "search_ablation_summary.txt").read_text())
    print(f"Raw:      {args.output_dir / 'search_ablation_raw.csv'}")
    print(f"Summary:  {args.output_dir / 'search_ablation_summary.csv'}")
    print(f"Pairwise: {args.output_dir / 'search_ablation_pairwise.csv'}")
    print(f"SF cache: {cache_path}")


if __name__ == "__main__":
    main()

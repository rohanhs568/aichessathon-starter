"""Deep pathological-position search diagnosis for the Chessathon V1 engine.

This pipeline is designed to run unattended for several hours.

It does four things in one run:

1. BUILD A POSITION POOL
   - positions from every PGN in the repository (unless excluded),
   - the existing hand-curated FEN suite,
   - a phase-stratified sample of the permanent Stockfish validation CSV,
   - any previously extracted regression CSVs.

2. DISCOVER REAL BASELINE FAILURES
   - run the *current search-lab baseline* at fixed depth 5,
   - evaluate Stockfish's best move and the baseline move from the ORIGINAL
     root position,
   - use root-player WDL expectation loss as the primary pathology metric,
     with CP loss as a secondary diagnostic,
   - confirm the strongest candidates again with the actual tournament time
     manager before they enter the pathology set.

3. DEEPLY ABLATE EACH PATHOLOGY
   At equal nominal depth, compare:
       baseline
       no_null / null_safe
       no_lmr / lmr_safe
       no_futility
       no_null_no_lmr
       no_null_no_futility
       no_lmr_no_futility
       no_selectivity
       capture_material_order
       no_aspiration
       no_pvs

   Each variant is searched iteratively through a depth ladder.  Every UNIQUE
   move that appears at any completed depth is then evaluated by deeper
   Stockfish with root_moves=[move].  This lets us identify *the depth at which
   a mechanism starts or stops corrupting the move*, instead of merely ranking
   final moves.

4. ATTRIBUTE THE FAILURE
   For every case the report asks, in order:
       - Does deeper baseline search fix it?               -> horizon/depth
       - Does removing one mechanism fix it at same depth? -> single mechanism
       - Does only a pair removal fix it?                  -> interaction
       - Does all selectivity off fix it?                  -> selectivity family
       - Does capture ordering fix it?                     -> ordering/horizon
       - Does PVS/aspiration removal fix it?                -> re-search instability
       - Does the NN one-ply static score prefer bad move? -> evaluator suspicion
       - Does qsearch prefer bad move?                     -> qsearch/horizon suspicion
       - Otherwise                                         -> unresolved/mixed

All Stockfish comparisons are made from the player-at-root's perspective.
The candidate is analysed on the ORIGINAL root with root_moves=[candidate].
That avoids the sign mistake caused by pushing the move and then reading a
side-to-move score.

The primary quality metric is Stockfish WDL expectation loss. CP loss remains
in every CSV because it is intuitive around balanced positions, but CP is not
linear in game outcome and mates need special handling.
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
import statistics
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import chess
import chess.engine
import chess.pgn


# ---------------------------------------------------------------------------
# Constants / experiment definition
# ---------------------------------------------------------------------------

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

SINGLE_VARIANTS = [
    "no_null",
    "null_safe",
    "no_lmr",
    "lmr_safe",
    "no_futility",
    "capture_material_order",
    "no_aspiration",
    "no_pvs",
]

PAIR_VARIANTS = [
    "no_null_no_lmr",
    "no_null_no_futility",
    "no_lmr_no_futility",
]

DEEP_VARIANTS = [
    "baseline",
    *SINGLE_VARIANTS,
    *PAIR_VARIANTS,
    "no_selectivity",
]

# When comparing a variant to the baseline at the SAME fixed depth, call it a
# meaningful rescue if either WDL expectation improves by this amount OR CP
# loss improves this much.  The WDL threshold is primary.
RESCUE_EXP = 0.04
RESCUE_CP = 100.0

# Baseline pathology thresholds.  We intentionally use an OR: a position can
# be important because of a large CP tactical miss even when WDL is compressed,
# or because of a large win-probability swing in a near-equal position.
PATH_EXP = 0.06
PATH_CP = 120.0


@dataclass
class PositionCase:
    case_id: str
    fen: str
    source: str
    source_kind: str
    category: str
    game_index: int | None = None
    ply: int | None = None
    played_move: str = ""


class BudgetExpired(Exception):
    pass


class WallBudget:
    def __init__(self, hours: float, reserve_minutes: float = 8.0):
        self.started = time.monotonic()
        self.deadline = self.started + hours * 3600.0
        self.reserve = reserve_minutes * 60.0

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def check(self, need_seconds: float = 0.0):
        if self.remaining() < self.reserve + need_seconds:
            raise BudgetExpired

    def status(self) -> str:
        return (
            f"elapsed={self.elapsed()/3600:.2f}h "
            f"remaining={max(0.0, self.remaining())/3600:.2f}h"
        )


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
    tmp.replace(path)


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return default


def position_identity_from_fen(fen: str) -> tuple:
    """Dedup identity suitable for an experiment pool.

    Search result can depend on the rule-50 counter, so unlike a pure
    repetition key we keep halfmove_clock here.  Fullmove number is irrelevant.
    """
    board = chess.Board(fen)
    return (
        board.board_fen(),
        board.turn,
        int(board.castling_rights),
        board.ep_square,
        int(board.halfmove_clock),
    )


def phase_category(board: chess.Board) -> str:
    n = len(board.piece_map())
    if n <= 10:
        return "endgame_2_10"
    if n <= 16:
        return "late_11_16"
    if n <= 23:
        return "middle_17_23"
    return "high_material_24_32"


def tactical_category(board: chess.Board) -> str:
    if board.is_check():
        return "in_check"
    captures = sum(1 for _ in board.generate_legal_captures())
    if captures >= 4:
        return "capture_rich"
    if captures:
        return "capture_available"
    return "quiet"


def material_balance(board: chess.Board, color: chess.Color) -> int:
    return sum(
        value * (
            len(board.pieces(piece, color))
            - len(board.pieces(piece, not color))
        )
        for piece, value in PIECE_VALUE.items()
    )


def case_priority(row: dict) -> float:
    """Sort pathologies by WDL first, then CP, with mate/sign failures boosted."""
    score = 10.0 * float(row.get("expectation_loss", 0.0))
    score += min(float(row.get("cp_loss", 0.0)), 1000.0) / 500.0
    if row.get("mate_blunder"):
        score += 5.0
    if row.get("sign_flip"):
        score += 1.0
    if row.get("winning_to_notwinning"):
        score += 1.0
    return score


# ---------------------------------------------------------------------------
# Search-lab loading and dynamic variants
# ---------------------------------------------------------------------------


def load_lab_agent(path: Path, weights: Path):
    os.environ["SEARCHLAB_MODEL_PATH"] = str(weights.resolve())
    spec = importlib.util.spec_from_file_location("deep_pathology_searchlab", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    with redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)

    required = {
        "baseline",
        "no_null",
        "null_safe",
        "no_lmr",
        "lmr_safe",
        "no_futility",
        "no_selectivity",
        "capture_material_order",
        "no_aspiration",
        "no_pvs",
    }
    missing = required - set(module.SEARCHLAB_PRESETS)
    if missing:
        raise RuntimeError(
            "searchlab_agent.py is missing required presets: "
            + ", ".join(sorted(missing))
        )

    # Pairwise ablations let us identify interactions rather than concluding
    # "all pruning" when two individually harmless mechanisms compose badly.
    module.SEARCHLAB_PRESETS.update(
        {
            "no_null_no_lmr": {
                "enable_null": False,
                "enable_lmr": False,
            },
            "no_null_no_futility": {
                "enable_null": False,
                "enable_futility": False,
            },
            "no_lmr_no_futility": {
                "enable_lmr": False,
                "enable_futility": False,
            },
        }
    )
    return module


def prepare_search(agent, board: chess.Board, variant: str, cap_seconds: float):
    agent.set_searchlab_variant(variant)
    agent.reset_searchlab_state(reset_game=True)
    agent._GAME_COUNTS = agent._build_game_counts(board)
    agent._PATH_COUNTS = {}
    agent.DEADLINE = time.monotonic() + cap_seconds
    agent.NODES = 0


def fixed_depth_trace(
    agent,
    fen: str,
    variant: str,
    max_depth: int,
    cap_seconds: float,
) -> dict:
    """One iterative-deepening run; keep every completed depth.

    This is more efficient and more diagnostic than restarting depth 1..D for
    each point on the depth curve. TT/history are allowed to behave exactly as
    iterative deepening normally uses them.
    """
    board = chess.Board(fen)
    prepare_search(agent, board, variant, cap_seconds)

    trace = []
    best_score = None
    timeout = False
    started = time.perf_counter()

    for depth in range(1, max_depth + 1):
        before_nodes = agent.NODES
        before_time = time.perf_counter()
        try:
            move, score = agent.aspiration_search(board, depth, best_score)
        except agent.SearchTimeout:
            timeout = True
            break
        if move is None:
            break
        best_score = score
        trace.append(
            {
                "depth": depth,
                "move": move.uci(),
                "engine_score_cp": int(score),
                "depth_nodes": int(agent.NODES - before_nodes),
                "total_nodes": int(agent.NODES),
                "depth_elapsed_s": time.perf_counter() - before_time,
                "total_elapsed_s": time.perf_counter() - started,
            }
        )

    return {
        "variant": variant,
        "trace": trace,
        "completed_depth": trace[-1]["depth"] if trace else 0,
        "move": trace[-1]["move"] if trace else "",
        "engine_score_cp": trace[-1]["engine_score_cp"] if trace else None,
        "nodes": int(agent.NODES),
        "elapsed_s": time.perf_counter() - started,
        "timeout": timeout,
        "stats": agent.get_searchlab_stats(),
    }


def timed_search(agent, fen: str, variant: str, time_left_ms: int) -> dict:
    board = chess.Board(fen)
    agent.set_searchlab_variant(variant)
    agent.reset_searchlab_state(reset_game=True)

    capture = io.StringIO()
    started = time.perf_counter()
    with redirect_stdout(capture):
        move = agent.get_move(fen, time_left_ms)
    elapsed = time.perf_counter() - started

    # The search-lab agent exposes final NODES. completed_depth is parsed from
    # its stable summary line, but a missing line is not fatal.
    completed_depth = 0
    score = None
    for line in capture.getvalue().splitlines():
        if line.startswith("depth="):
            parts = dict(
                item.split("=", 1)
                for item in line.split()
                if "=" in item
            )
            try:
                completed_depth = max(completed_depth, int(parts.get("depth", 0)))
                score = int(parts.get("score"))
            except Exception:
                pass

    legal = chess.Move.from_uci(move)
    if legal not in board.legal_moves:
        raise RuntimeError(f"{variant} returned illegal move {move}")

    return {
        "variant": variant,
        "move": move,
        "completed_depth": completed_depth,
        "engine_score_cp": score,
        "nodes": int(agent.NODES),
        "elapsed_s": elapsed,
        "stats": agent.get_searchlab_stats(),
    }


def static_move_score(agent, fen: str, move_uci: str) -> int | None:
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        return None
    board.push(move)
    return -int(agent.evaluate(board))


def qsearch_move_score(agent, fen: str, move_uci: str, cap_seconds: float = 4.0):
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        return None
    agent.set_searchlab_variant("baseline")
    agent.reset_searchlab_state(reset_game=True)
    agent._GAME_COUNTS = agent._build_game_counts(board)
    agent._PATH_COUNTS = {}
    agent.DEADLINE = time.monotonic() + cap_seconds
    agent.NODES = 0
    board.push(move)
    try:
        return -int(agent.quiescence(board, -agent.INF, agent.INF, 1))
    except agent.SearchTimeout:
        return None
    finally:
        board.pop()


# ---------------------------------------------------------------------------
# Stockfish reference measurement
# ---------------------------------------------------------------------------


def configure_stockfish(engine, hash_mb: int):
    opts = {}
    if "Threads" in engine.options:
        opts["Threads"] = 1
    if "Hash" in engine.options:
        opts["Hash"] = hash_mb
    if "UCI_ShowWDL" in engine.options:
        opts["UCI_ShowWDL"] = True
    if opts:
        engine.configure(opts)


def score_record(info: dict, root_color: chess.Color, ply: int) -> dict:
    pov = info["score"].pov(root_color)
    cp_equiv = pov.score(mate_score=100_000)
    mate = pov.mate()

    if "wdl" in info:
        wdl = info["wdl"].pov(root_color)
    else:
        wdl = pov.wdl(model="sf", ply=ply)

    return {
        "score_text": str(pov),
        "cp_equiv": int(cp_equiv) if cp_equiv is not None else None,
        "mate": mate,
        "is_mate": bool(pov.is_mate()),
        "expectation": float(wdl.expectation()),
        "wdl_wins": int(wdl.wins),
        "wdl_draws": int(wdl.draws),
        "wdl_losses": int(wdl.losses),
        "pv": [m.uci() for m in info.get("pv", [])[:20]],
        "depth": int(info.get("depth", 0)),
        "seldepth": int(info.get("seldepth", 0)),
        "nodes": int(info.get("nodes", 0)),
    }


def sf_best(engine, board, depth, cache, cache_path) -> dict:
    key = f"best|d={depth}|{board.fen()}"
    if key not in cache:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        record = score_record(info, board.turn, board.ply())
        record["best_move"] = record["pv"][0] if record["pv"] else None
        cache[key] = record
        save_json(cache_path, cache)
    return cache[key]


def sf_forced(engine, board, move_uci, depth, cache, cache_path, best=None) -> dict:
    if best is not None and best.get("best_move") == move_uci:
        return best
    key = f"move|d={depth}|{move_uci}|{board.fen()}"
    if key not in cache:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            raise RuntimeError(f"illegal candidate {move_uci} for {board.fen()}")
        info = engine.analyse(
            board,
            chess.engine.Limit(depth=depth),
            root_moves=[move],
        )
        record = score_record(info, board.turn, board.ply())
        record["best_move"] = move_uci
        cache[key] = record
        save_json(cache_path, cache)
    return cache[key]


def quality_delta(best: dict, candidate: dict) -> dict:
    best_cp = float(best["cp_equiv"])
    cand_cp = float(candidate["cp_equiv"])
    cp_raw = best_cp - cand_cp
    cp_loss = max(0.0, cp_raw)
    exp_raw = float(best["expectation"]) - float(candidate["expectation"])
    exp_loss = max(0.0, exp_raw)

    mate_blunder = bool(
        candidate.get("mate") is not None
        and candidate["mate"] < 0
        and not (best.get("mate") is not None and best["mate"] < 0)
    )
    sign_flip = bool(best_cp >= 100 and cand_cp <= -100)
    winning_to_notwinning = bool(
        best["expectation"] >= 0.75 and candidate["expectation"] < 0.55
    )

    return {
        "cp_delta_raw": cp_raw,
        "cp_loss": cp_loss,
        "expectation_delta_raw": exp_raw,
        "expectation_loss": exp_loss,
        "mate_blunder": mate_blunder,
        "sign_flip": sign_flip,
        "winning_to_notwinning": winning_to_notwinning,
    }


# ---------------------------------------------------------------------------
# Position pool construction
# ---------------------------------------------------------------------------


def skip_path(path: Path) -> bool:
    bad_parts = {
        ".git",
        ".venv",
        ".venv-gpu",
        "__pycache__",
        "training/models",
        "deep_pathology_results",
    }
    s = str(path)
    return any(part in s for part in bad_parts)


def find_pgns(repo: Path) -> list[Path]:
    result = []
    for path in repo.rglob("*.pgn"):
        if skip_path(path):
            continue
        if path.is_file():
            result.append(path)
    return sorted(set(result))


def positions_from_pgns(paths: list[Path], max_positions: int, rng: random.Random):
    pool: list[PositionCase] = []
    for path in paths:
        try:
            with path.open(errors="replace") as f:
                game_idx = 0
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    game_idx += 1
                    board = game.board()
                    ply = 0
                    for move in game.mainline_moves():
                        ply += 1
                        # Skip only terminal/degenerate states. Keep early positions;
                        # the direct baseline confirmation decides usefulness.
                        if not board.is_game_over(claim_draw=True):
                            pool.append(
                                PositionCase(
                                    case_id=f"pgn_{path.stem}_g{game_idx}_p{ply}",
                                    fen=board.fen(),
                                    source=str(path),
                                    source_kind="pgn",
                                    category=(
                                        phase_category(board)
                                        + "+"
                                        + tactical_category(board)
                                    ),
                                    game_index=game_idx,
                                    ply=ply,
                                    played_move=move.uci(),
                                )
                            )
                        board.push(move)
        except Exception as exc:
            print(f"warning: could not parse {path}: {exc!r}", flush=True)

    if len(pool) > max_positions:
        pool = rng.sample(pool, max_positions)
    return pool


def positions_from_fen_file(path: Path) -> list[PositionCase]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for i, raw in enumerate(f, start=1):
            fen = raw.strip()
            if not fen or fen.startswith("#"):
                continue
            try:
                board = chess.Board(fen)
            except Exception:
                continue
            out.append(
                PositionCase(
                    case_id=f"suite_{i:04d}",
                    fen=fen,
                    source=str(path),
                    source_kind="fen_suite",
                    category=phase_category(board) + "+" + tactical_category(board),
                )
            )
    return out


def positions_from_regression_csv(path: Path) -> list[PositionCase]:
    if not path.exists():
        return []
    out = []
    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=1):
                fen = (row.get("fen") or "").strip()
                if not fen:
                    continue
                board = chess.Board(fen)
                out.append(
                    PositionCase(
                        case_id=row.get("id") or row.get("case_id") or f"reg_{i:04d}",
                        fen=fen,
                        source=str(path),
                        source_kind="regression_csv",
                        category=row.get("category") or (
                            phase_category(board) + "+" + tactical_category(board)
                        ),
                        played_move=row.get("played_move") or "",
                    )
                )
    except Exception as exc:
        print(f"warning: failed regression CSV {path}: {exc!r}")
    return out


def piece_count_from_fen_text(fen: str) -> int:
    board_field = fen.split()[0]
    return sum(1 for c in board_field if c.isalpha())


def validation_phase_samples(path: Path, per_phase: int, rng: random.Random):
    """Reservoir-sample each material phase from the permanent validation set."""
    if not path.exists():
        return []

    buckets = {
        "endgame_2_10": [],
        "late_11_16": [],
        "middle_17_23": [],
        "high_material_24_32": [],
    }
    seen = {k: 0 for k in buckets}

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "fen" not in (reader.fieldnames or []):
            return []
        for row in reader:
            fen = row["fen"]
            n = piece_count_from_fen_text(fen)
            if n <= 10:
                b = "endgame_2_10"
            elif n <= 16:
                b = "late_11_16"
            elif n <= 23:
                b = "middle_17_23"
            else:
                b = "high_material_24_32"
            seen[b] += 1
            reservoir = buckets[b]
            if len(reservoir) < per_phase:
                reservoir.append(fen)
            else:
                j = rng.randrange(seen[b])
                if j < per_phase:
                    reservoir[j] = fen

    out = []
    idx = 0
    for phase, fens in buckets.items():
        for fen in fens:
            try:
                board = chess.Board(fen)
            except Exception:
                continue
            idx += 1
            out.append(
                PositionCase(
                    case_id=f"validation_{idx:04d}",
                    fen=fen,
                    source=str(path),
                    source_kind="validation_sample",
                    category=phase + "+" + tactical_category(board),
                )
            )
    return out


def dedup_cases(cases: Iterable[PositionCase], max_pool: int, rng: random.Random):
    by_key = {}
    # Prefer actual PGN / regression cases when duplicate positions occur.
    priority = {"regression_csv": 4, "pgn": 3, "fen_suite": 2, "validation_sample": 1}
    for case in cases:
        try:
            key = position_identity_from_fen(case.fen)
        except Exception:
            continue
        prev = by_key.get(key)
        if prev is None or priority.get(case.source_kind, 0) > priority.get(prev.source_kind, 0):
            by_key[key] = case

    values = list(by_key.values())
    if len(values) <= max_pool:
        return values

    # Keep all rare endgames/regression cases where possible, sample the rest.
    must = [
        c for c in values
        if c.source_kind == "regression_csv" or c.category.startswith("endgame")
    ]
    must = must[: max_pool // 2]
    must_keys = {position_identity_from_fen(c.fen) for c in must}
    rest = [c for c in values if position_identity_from_fen(c.fen) not in must_keys]
    take = max(0, max_pool - len(must))
    if len(rest) > take:
        rest = rng.sample(rest, take)
    return must + rest


# ---------------------------------------------------------------------------
# Discovery and confirmation
# ---------------------------------------------------------------------------


def discovery_scan(
    cases,
    agent,
    engine,
    cache,
    cache_path,
    out_dir,
    budget,
    screen_agent_depth,
    screen_sf_depth,
    search_cap_s,
):
    rows = []
    path = out_dir / "01_discovery_all.csv"

    print(
        f"\nDISCOVERY: {len(cases)} positions, baseline fixed-depth {screen_agent_depth}, "
        f"SF depth {screen_sf_depth}",
        flush=True,
    )

    for i, case in enumerate(cases, start=1):
        budget.check(need_seconds=search_cap_s + 3.0)
        board = chess.Board(case.fen)
        try:
            search = fixed_depth_trace(
                agent,
                case.fen,
                "baseline",
                screen_agent_depth,
                search_cap_s,
            )
            if not search["move"]:
                continue
            best = sf_best(engine, board, screen_sf_depth, cache, cache_path)
            cand = sf_forced(
                engine,
                board,
                search["move"],
                screen_sf_depth,
                cache,
                cache_path,
                best,
            )
            delta = quality_delta(best, cand)
            row = {
                **asdict(case),
                "baseline_move": search["move"],
                "completed_depth": search["completed_depth"],
                "agent_nodes": search["nodes"],
                "agent_elapsed_s": search["elapsed_s"],
                "sf_best_move": best.get("best_move"),
                "sf_best_score": best["score_text"],
                "sf_candidate_score": cand["score_text"],
                "sf_best_expectation": best["expectation"],
                "sf_candidate_expectation": cand["expectation"],
                **delta,
            }
            row["pathology_priority"] = case_priority(row)
            rows.append(row)
            if i % 10 == 0 or row["expectation_loss"] >= PATH_EXP or row["cp_loss"] >= PATH_CP:
                tag = "PATH" if (row["expectation_loss"] >= PATH_EXP or row["cp_loss"] >= PATH_CP) else ""
                print(
                    f"[{i:4d}/{len(cases)}] {tag:<4} {case.case_id:<28} "
                    f"move={search['move']} loss={row['cp_loss']:.0f}cp "
                    f"exp={row['expectation_loss']:.3f} d={search['completed_depth']} "
                    f"{budget.status()}",
                    flush=True,
                )
        except Exception as exc:
            rows.append({**asdict(case), "error": repr(exc)})
            print(f"[{i}/{len(cases)}] ERROR {case.case_id}: {exc!r}", flush=True)

        if i % 5 == 0:
            write_csv(path, rows)

    write_csv(path, rows)
    return rows


def select_screen_candidates(rows: list[dict], max_candidates: int):
    good = [r for r in rows if not r.get("error") and r.get("baseline_move")]
    severe = [
        r for r in good
        if r["expectation_loss"] >= PATH_EXP
        or r["cp_loss"] >= PATH_CP
        or r["mate_blunder"]
        or r["sign_flip"]
    ]
    severe.sort(key=case_priority, reverse=True)

    if len(severe) >= max_candidates:
        return severe[:max_candidates]

    # If the corpus happened to be easy, keep the strongest near-misses too.
    # This means the unattended run still diagnoses the tail rather than ending
    # with an empty experiment.
    selected = list(severe)
    used = {r["fen"] for r in selected}
    fallback = sorted(good, key=case_priority, reverse=True)
    for row in fallback:
        if row["fen"] in used:
            continue
        selected.append(row)
        used.add(row["fen"])
        if len(selected) >= max_candidates:
            break
    return selected


def confirm_timed_pathologies(
    screen_rows,
    agent,
    engine,
    cache,
    cache_path,
    out_dir,
    budget,
    time_left_ms,
    confirm_sf_depth,
    max_pathologies,
):
    rows = []
    print(
        f"\nCONFIRMATION: {len(screen_rows)} strongest screen cases at tournament time",
        flush=True,
    )

    for i, src in enumerate(screen_rows, start=1):
        budget.check(need_seconds=6.0)
        board = chess.Board(src["fen"])
        try:
            search = timed_search(agent, src["fen"], "baseline", time_left_ms)
            best = sf_best(engine, board, confirm_sf_depth, cache, cache_path)
            cand = sf_forced(
                engine, board, search["move"], confirm_sf_depth, cache, cache_path, best
            )
            delta = quality_delta(best, cand)
            row = {
                **src,
                "screen_baseline_move": src["baseline_move"],
                "screen_cp_loss": src["cp_loss"],
                "screen_expectation_loss": src["expectation_loss"],
                "baseline_move": search["move"],
                "completed_depth": search["completed_depth"],
                "agent_nodes": search["nodes"],
                "agent_elapsed_s": search["elapsed_s"],
                "sf_best_move": best.get("best_move"),
                "sf_best_score": best["score_text"],
                "sf_candidate_score": cand["score_text"],
                "sf_best_expectation": best["expectation"],
                "sf_candidate_expectation": cand["expectation"],
                **delta,
            }
            row["pathology_priority"] = case_priority(row)
            rows.append(row)
            print(
                f"[{i:2d}/{len(screen_rows)}] {src['case_id']:<28} "
                f"timed={search['move']} loss={row['cp_loss']:.0f}cp "
                f"exp={row['expectation_loss']:.3f} d={search['completed_depth']}",
                flush=True,
            )
        except Exception as exc:
            rows.append({**src, "confirm_error": repr(exc)})
            print(f"confirm ERROR {src['case_id']}: {exc!r}", flush=True)

        write_csv(out_dir / "02_timed_confirmation_all.csv", rows)

    valid = [r for r in rows if not r.get("confirm_error")]
    true_paths = [
        r for r in valid
        if r["expectation_loss"] >= PATH_EXP
        or r["cp_loss"] >= PATH_CP
        or r["mate_blunder"]
        or r["sign_flip"]
    ]
    true_paths.sort(key=case_priority, reverse=True)

    # If fewer than requested survive exact time control, include the most
    # pathological timed near-misses. They remain useful tail diagnostics but
    # are labelled as such.
    selected = true_paths[:max_pathologies]
    used = {r["fen"] for r in selected}
    if len(selected) < max_pathologies:
        for row in sorted(valid, key=case_priority, reverse=True):
            if row["fen"] in used:
                continue
            row = dict(row)
            row["near_miss_only"] = True
            selected.append(row)
            used.add(row["fen"])
            if len(selected) >= max_pathologies:
                break

    write_csv(out_dir / "03_pathology_set.csv", selected)
    with (out_dir / "03_pathology_fens.txt").open("w") as f:
        for row in selected:
            f.write(row["fen"] + "\n")

    print(
        f"\nSelected {len(selected)} pathology/tail cases for deep diagnosis. "
        f"True threshold cases: {len(true_paths)}",
        flush=True,
    )
    return selected


# ---------------------------------------------------------------------------
# Deep fixed-depth mechanism diagnosis
# ---------------------------------------------------------------------------


def deep_ablation(
    cases,
    agent,
    engine,
    cache,
    cache_path,
    out_dir,
    budget,
    deep_sf_depth,
    max_agent_depth,
    per_search_cap_s,
):
    rows = []
    raw_path = out_dir / "04_deep_ablation_raw.csv"

    print(
        f"\nDEEP ABLATION: {len(cases)} cases x {len(DEEP_VARIANTS)} variants, "
        f"depth ladder 1..{max_agent_depth}, per-search cap {per_search_cap_s:g}s",
        flush=True,
    )

    for case_idx, case in enumerate(cases, start=1):
        budget.check(need_seconds=30.0)
        board = chess.Board(case["fen"])
        best = sf_best(engine, board, deep_sf_depth, cache, cache_path)
        print(
            f"\n[{case_idx}/{len(cases)}] {case['case_id']} "
            f"SF={best.get('best_move')} {best['score_text']} exp={best['expectation']:.3f}",
            flush=True,
        )

        for variant in DEEP_VARIANTS:
            budget.check(need_seconds=per_search_cap_s + 2.0)
            try:
                run = fixed_depth_trace(
                    agent,
                    case["fen"],
                    variant,
                    max_agent_depth,
                    per_search_cap_s,
                )
                if not run["trace"]:
                    rows.append(
                        {
                            "case_id": case["case_id"],
                            "fen": case["fen"],
                            "variant": variant,
                            "error": "no completed depth",
                        }
                    )
                    continue

                # Only unique moves need a deep Stockfish forced analysis.
                forced_by_move = {}
                for point in run["trace"]:
                    move_uci = point["move"]
                    if move_uci not in forced_by_move:
                        forced_by_move[move_uci] = sf_forced(
                            engine,
                            board,
                            move_uci,
                            deep_sf_depth,
                            cache,
                            cache_path,
                            best,
                        )

                    forced = forced_by_move[move_uci]
                    delta = quality_delta(best, forced)
                    row = {
                        "case_id": case["case_id"],
                        "fen": case["fen"],
                        "category": case.get("category", ""),
                        "source": case.get("source", ""),
                        "variant": variant,
                        **point,
                        "run_completed_depth": run["completed_depth"],
                        "run_timeout": run["timeout"],
                        "run_elapsed_s": run["elapsed_s"],
                        "sf_best_move": best.get("best_move"),
                        "sf_best_score": best["score_text"],
                        "sf_candidate_score": forced["score_text"],
                        "sf_best_cp": best["cp_equiv"],
                        "sf_candidate_cp": forced["cp_equiv"],
                        "sf_best_expectation": best["expectation"],
                        "sf_candidate_expectation": forced["expectation"],
                        **delta,
                    }
                    for stat, value in run["stats"].items():
                        if stat != "variant":
                            row[f"stat_{stat}"] = value
                    rows.append(row)

                final = rows[-1]
                print(
                    f"  {variant:<24} d={run['completed_depth']} move={run['move']} "
                    f"loss={final['cp_loss']:.0f}cp exp={final['expectation_loss']:.3f} "
                    f"nodes={run['nodes']} timeout={run['timeout']}",
                    flush=True,
                )
            except BudgetExpired:
                raise
            except Exception as exc:
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "fen": case["fen"],
                        "variant": variant,
                        "error": repr(exc),
                    }
                )
                print(f"  {variant:<24} ERROR {exc!r}", flush=True)

            write_csv(raw_path, rows)

    write_csv(raw_path, rows)
    return rows


# ---------------------------------------------------------------------------
# Static / qsearch diagnostics and attribution
# ---------------------------------------------------------------------------


def latest_at_depth(rows: list[dict], case_id: str, variant: str, depth: int):
    matches = [
        r for r in rows
        if r.get("case_id") == case_id
        and r.get("variant") == variant
        and int(r.get("depth", -1)) == depth
        and not r.get("error")
    ]
    return matches[-1] if matches else None


def deepest_row(rows: list[dict], case_id: str, variant: str):
    matches = [
        r for r in rows
        if r.get("case_id") == case_id
        and r.get("variant") == variant
        and not r.get("error")
    ]
    if not matches:
        return None
    return max(matches, key=lambda r: int(r.get("depth", -1)))


def is_bad(row: dict | None) -> bool:
    if row is None:
        return True
    return (
        float(row.get("expectation_loss", 0.0)) >= PATH_EXP
        or float(row.get("cp_loss", 0.0)) >= PATH_CP
        or bool(row.get("mate_blunder"))
        or bool(row.get("sign_flip"))
    )


def rescue_amount(base: dict, other: dict):
    d_exp = float(base["expectation_loss"]) - float(other["expectation_loss"])
    d_cp = float(base["cp_loss"]) - float(other["cp_loss"])
    rescued = d_exp >= RESCUE_EXP or d_cp >= RESCUE_CP
    return rescued, d_exp, d_cp


def static_q_diagnostics(cases, deep_rows, agent, out_dir, budget):
    result = []
    for case in cases:
        budget.check(need_seconds=8.0)
        cid = case["case_id"]
        baseline = deepest_row(deep_rows, cid, "baseline")
        if baseline is None:
            continue
        sf_move = baseline.get("sf_best_move")
        bad_move = baseline.get("move")
        if not sf_move or not bad_move:
            continue

        board = chess.Board(case["fen"])
        before_material = material_balance(board, board.turn)

        def mat_after(move_uci):
            b = chess.Board(case["fen"])
            root = b.turn
            move = chess.Move.from_uci(move_uci)
            if move not in b.legal_moves:
                return None
            b.push(move)
            return material_balance(b, root)

        row = {
            "case_id": cid,
            "fen": case["fen"],
            "sf_best_move": sf_move,
            "baseline_deep_move": bad_move,
            "root_material_cp": before_material,
            "sf_move_material_after_cp": mat_after(sf_move),
            "baseline_move_material_after_cp": mat_after(bad_move),
            "sf_move_nn_static_cp": static_move_score(agent, case["fen"], sf_move),
            "baseline_move_nn_static_cp": static_move_score(agent, case["fen"], bad_move),
            "sf_move_qsearch_cp": qsearch_move_score(agent, case["fen"], sf_move),
            "baseline_move_qsearch_cp": qsearch_move_score(agent, case["fen"], bad_move),
        }
        if row["sf_move_nn_static_cp"] is not None and row["baseline_move_nn_static_cp"] is not None:
            row["nn_static_bad_minus_best"] = (
                row["baseline_move_nn_static_cp"] - row["sf_move_nn_static_cp"]
            )
        if row["sf_move_qsearch_cp"] is not None and row["baseline_move_qsearch_cp"] is not None:
            row["qsearch_bad_minus_best"] = (
                row["baseline_move_qsearch_cp"] - row["sf_move_qsearch_cp"]
            )
        result.append(row)
        write_csv(out_dir / "05_static_qsearch_diagnostics.csv", result)
    return result


def attribute_cases(cases, deep_rows, static_rows, primary_depth: int):
    static_by_id = {r["case_id"]: r for r in static_rows}
    attributions = []

    for case in cases:
        cid = case["case_id"]
        baseline_d = latest_at_depth(deep_rows, cid, "baseline", primary_depth)
        if baseline_d is None:
            baseline_d = deepest_row(deep_rows, cid, "baseline")
        if baseline_d is None:
            continue
        depth_used = int(baseline_d["depth"])

        singles = []
        for variant in SINGLE_VARIANTS:
            other = latest_at_depth(deep_rows, cid, variant, depth_used)
            if other is None:
                continue
            rescued, d_exp, d_cp = rescue_amount(baseline_d, other)
            if rescued:
                singles.append((variant, d_exp, d_cp, other))

        pairs = []
        for variant in PAIR_VARIANTS:
            other = latest_at_depth(deep_rows, cid, variant, depth_used)
            if other is None:
                continue
            rescued, d_exp, d_cp = rescue_amount(baseline_d, other)
            if rescued:
                pairs.append((variant, d_exp, d_cp, other))

        no_sel = latest_at_depth(deep_rows, cid, "no_selectivity", depth_used)
        no_sel_rescue = False
        no_sel_dexp = no_sel_dcp = 0.0
        if no_sel is not None:
            no_sel_rescue, no_sel_dexp, no_sel_dcp = rescue_amount(baseline_d, no_sel)

        # Does ordinary baseline solve its own problem by searching deeper?
        baseline_deep = deepest_row(deep_rows, cid, "baseline")
        deeper_rescue = False
        if baseline_deep is not None and int(baseline_deep["depth"]) > depth_used:
            deeper_rescue, _, _ = rescue_amount(baseline_d, baseline_deep)

        sdiag = static_by_id.get(cid, {})
        nn_prefers_bad = float(sdiag.get("nn_static_bad_minus_best") or 0.0) >= 100.0
        q_prefers_bad = float(sdiag.get("qsearch_bad_minus_best") or 0.0) >= 100.0

        singles_sorted = sorted(singles, key=lambda x: (x[1], x[2]), reverse=True)
        pairs_sorted = sorted(pairs, key=lambda x: (x[1], x[2]), reverse=True)

        if singles_sorted:
            primary = f"single:{singles_sorted[0][0]}"
            confidence = "high" if singles_sorted[0][1] >= 0.08 or singles_sorted[0][2] >= 180 else "medium"
        elif pairs_sorted:
            primary = f"interaction:{pairs_sorted[0][0]}"
            confidence = "medium"
        elif no_sel_rescue:
            primary = "selectivity:higher_order"
            confidence = "medium"
        elif deeper_rescue:
            primary = "depth_horizon"
            confidence = "medium"
        elif q_prefers_bad and not nn_prefers_bad:
            primary = "qsearch_horizon_suspect"
            confidence = "medium"
        elif nn_prefers_bad:
            primary = "evaluator_static_suspect"
            confidence = "medium"
        else:
            primary = "unresolved_mixed_or_depth"
            confidence = "low"

        attributions.append(
            {
                "case_id": cid,
                "fen": case["fen"],
                "category": case.get("category", ""),
                "primary_depth": depth_used,
                "baseline_move": baseline_d["move"],
                "sf_best_move": baseline_d["sf_best_move"],
                "baseline_cp_loss": baseline_d["cp_loss"],
                "baseline_expectation_loss": baseline_d["expectation_loss"],
                "attribution": primary,
                "confidence": confidence,
                "single_rescues": ";".join(x[0] for x in singles_sorted),
                "pair_rescues": ";".join(x[0] for x in pairs_sorted),
                "no_selectivity_rescue": no_sel_rescue,
                "deeper_baseline_rescue": deeper_rescue,
                "nn_prefers_bad_by_100cp": nn_prefers_bad,
                "qsearch_prefers_bad_by_100cp": q_prefers_bad,
                "nn_static_bad_minus_best": sdiag.get("nn_static_bad_minus_best"),
                "qsearch_bad_minus_best": sdiag.get("qsearch_bad_minus_best"),
                "no_selectivity_dExp": no_sel_dexp,
                "no_selectivity_dCP": no_sel_dcp,
            }
        )

    return attributions


# ---------------------------------------------------------------------------
# Competition-time re-test of attributed mechanisms
# ---------------------------------------------------------------------------


def choose_timed_variants(attribution: dict):
    variants = ["baseline"]
    attr = attribution["attribution"]
    if attr.startswith("single:"):
        variants.append(attr.split(":", 1)[1])
    elif attr.startswith("interaction:"):
        variants.append(attr.split(":", 1)[1])
    elif attr.startswith("selectivity:"):
        variants.append("no_selectivity")
    elif attr == "depth_horizon":
        # Time control cannot force an extra depth, but no-selectivity is a useful
        # contrast for whether pruning helps/hurts the real iteration depth.
        variants.append("no_selectivity")
    elif attr == "qsearch_horizon_suspect":
        variants.append("capture_material_order")
    elif attr == "evaluator_static_suspect":
        variants.extend(["no_selectivity", "no_lmr"])
    else:
        variants.extend(["no_selectivity", "no_lmr", "no_null"])

    # Always include the two most plausible global suspects, unless duplicated.
    variants.extend(["no_null", "no_lmr"])
    return list(dict.fromkeys(variants))


def timed_attribution_retest(
    cases,
    attributions,
    agent,
    engine,
    cache,
    cache_path,
    out_dir,
    budget,
    sf_depth,
    time_left_ms,
    max_cases,
):
    by_case = {a["case_id"]: a for a in attributions}
    case_by_id = {c["case_id"]: c for c in cases}
    selected = sorted(
        attributions,
        key=lambda a: (
            float(a["baseline_expectation_loss"]),
            float(a["baseline_cp_loss"]),
        ),
        reverse=True,
    )[:max_cases]

    rows = []
    print(f"\nTIMED RETEST: {len(selected)} strongest attributed cases", flush=True)

    for i, attr in enumerate(selected, start=1):
        case = case_by_id[attr["case_id"]]
        board = chess.Board(case["fen"])
        best = sf_best(engine, board, sf_depth, cache, cache_path)
        variants = choose_timed_variants(attr)
        print(
            f"[{i}/{len(selected)}] {attr['case_id']} attr={attr['attribution']} variants={','.join(variants)}",
            flush=True,
        )
        for variant in variants:
            budget.check(need_seconds=6.0)
            try:
                run = timed_search(agent, case["fen"], variant, time_left_ms)
                forced = sf_forced(
                    engine, board, run["move"], sf_depth, cache, cache_path, best
                )
                delta = quality_delta(best, forced)
                row = {
                    "case_id": attr["case_id"],
                    "fen": case["fen"],
                    "attribution": attr["attribution"],
                    "variant": variant,
                    "move": run["move"],
                    "completed_depth": run["completed_depth"],
                    "nodes": run["nodes"],
                    "elapsed_s": run["elapsed_s"],
                    "sf_best_move": best.get("best_move"),
                    "sf_best_score": best["score_text"],
                    "sf_candidate_score": forced["score_text"],
                    **delta,
                }
                rows.append(row)
                print(
                    f"  {variant:<24} {run['move']} d={run['completed_depth']} "
                    f"loss={row['cp_loss']:.0f}cp exp={row['expectation_loss']:.3f}",
                    flush=True,
                )
            except Exception as exc:
                rows.append(
                    {
                        "case_id": attr["case_id"],
                        "variant": variant,
                        "error": repr(exc),
                    }
                )
            write_csv(out_dir / "07_timed_attribution_retest.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def aggregate_attributions(rows: list[dict]):
    counts = {}
    for row in rows:
        key = row["attribution"]
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def make_report(
    out_dir,
    pool,
    discovery,
    pathology_cases,
    deep_rows,
    attributions,
    timed_rows,
    budget,
    args,
    stopped_early=False,
):
    lines = []
    lines.append("DEEP SEARCH PATHOLOGY REPORT")
    lines.append("=" * 88)
    lines.append(f"wall budget: {args.budget_hours:.2f} h")
    lines.append(f"actual elapsed: {budget.elapsed()/3600:.2f} h")
    lines.append(f"stopped early by budget: {stopped_early}")
    lines.append(f"position pool: {len(pool)}")
    lines.append(f"discovery positions completed: {len(discovery)}")
    lines.append(f"pathology/tail set: {len(pathology_cases)}")
    lines.append(f"deep ablation rows: {len(deep_rows)}")
    lines.append(f"attributed cases: {len(attributions)}")
    lines.append(f"timed retest rows: {len(timed_rows)}")
    lines.append("")

    lines.append("METRIC DEFINITION")
    lines.append("-" * 88)
    lines.append("Primary = Stockfish WDL expectation loss from the root player's POV.")
    lines.append("Secondary = root-player CP loss; mate is mapped to a large CP surrogate only for diagnostics.")
    lines.append("Every candidate is evaluated on the ORIGINAL root with root_moves=[candidate].")
    lines.append("")

    lines.append("ATTRIBUTION COUNTS")
    lines.append("-" * 88)
    for key, count in aggregate_attributions(attributions):
        lines.append(f"{count:3d}  {key}")
    if not attributions:
        lines.append("<none completed>")
    lines.append("")

    lines.append("CASE-BY-CASE")
    lines.append("-" * 88)
    ordered = sorted(
        attributions,
        key=lambda r: (float(r["baseline_expectation_loss"]), float(r["baseline_cp_loss"])),
        reverse=True,
    )
    for row in ordered:
        lines.append(
            f"{row['case_id']}: {row['attribution']} ({row['confidence']}) | "
            f"base {row['baseline_move']} loss={float(row['baseline_cp_loss']):.0f}cp/"
            f"{float(row['baseline_expectation_loss']):.3f} WDL | SF {row['sf_best_move']}"
        )
        if row.get("single_rescues"):
            lines.append(f"    single rescues: {row['single_rescues']}")
        if row.get("pair_rescues"):
            lines.append(f"    pair rescues:   {row['pair_rescues']}")
        if row.get("deeper_baseline_rescue"):
            lines.append("    deeper baseline itself rescues the position")
        if row.get("nn_prefers_bad_by_100cp"):
            lines.append(
                f"    NN static prefers bad move by {float(row.get('nn_static_bad_minus_best') or 0):.0f} cp"
            )
        if row.get("qsearch_prefers_bad_by_100cp"):
            lines.append(
                f"    qsearch prefers bad move by {float(row.get('qsearch_bad_minus_best') or 0):.0f} cp"
            )
    lines.append("")

    lines.append("INTERPRETATION RULES")
    lines.append("-" * 88)
    lines.append("single:X = removing X at the SAME depth materially rescues Stockfish move quality.")
    lines.append("interaction:A_B = singles did not rescue, but removing A+B together did.")
    lines.append("selectivity:higher_order = only broad selectivity removal rescued at same depth.")
    lines.append("depth_horizon = ordinary baseline gets materially better when searched deeper.")
    lines.append("evaluator_static_suspect = the one-ply NN itself ranks the bad move >=100cp above SF best.")
    lines.append("qsearch_horizon_suspect = static NN does not, but current qsearch does.")
    lines.append("unresolved = no tested mechanism cleanly explains the failure; inspect its depth/PV trace.")
    lines.append("")

    lines.append("FILES")
    lines.append("-" * 88)
    for name in [
        "00_position_pool.csv",
        "01_discovery_all.csv",
        "02_timed_confirmation_all.csv",
        "03_pathology_set.csv",
        "03_pathology_fens.txt",
        "04_deep_ablation_raw.csv",
        "05_static_qsearch_diagnostics.csv",
        "06_attribution.csv",
        "07_timed_attribution_retest.csv",
        "stockfish_cache.json",
    ]:
        lines.append(name)

    (out_dir / "FINAL_REPORT.txt").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--agent-lab", type=Path, default=Path("training/searchlab_agent.py"))
    parser.add_argument("--weights", type=Path, default=Path("weights/v1.npz"))
    parser.add_argument("--stockfish", default="/usr/games/stockfish")
    parser.add_argument("--validation", type=Path, default=Path("training/data/samples/lichess_validation_250k.csv"))
    parser.add_argument("--fen-suite", type=Path, default=Path("training/data/test_fens_60.txt"))
    parser.add_argument("--regression-csv", type=Path, default=Path("training/data/search_regression_cases.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("deep_pathology_results"))
    parser.add_argument("--budget-hours", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=568)

    parser.add_argument("--max-pool", type=int, default=700)
    parser.add_argument("--max-pgn-positions", type=int, default=250)
    parser.add_argument("--validation-per-phase", type=int, default=100)

    parser.add_argument("--screen-agent-depth", type=int, default=5)
    parser.add_argument("--screen-agent-cap-s", type=float, default=6.0)
    parser.add_argument("--screen-sf-depth", type=int, default=14)
    parser.add_argument("--confirm-sf-depth", type=int, default=18)
    parser.add_argument("--screen-candidates", type=int, default=50)
    parser.add_argument("--max-pathologies", type=int, default=18)

    parser.add_argument("--deep-sf-depth", type=int, default=22)
    parser.add_argument("--deep-agent-depth", type=int, default=7)
    parser.add_argument("--deep-search-cap-s", type=float, default=20.0)
    parser.add_argument("--time-left-ms", type=int, default=120000)
    parser.add_argument("--timed-retest-cases", type=int, default=10)
    parser.add_argument("--sf-hash", type=int, default=256)
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    budget = WallBudget(args.budget_hours)

    print("DEEP PATHOLOGY PIPELINE", flush=True)
    print("=" * 88, flush=True)
    print(f"repo: {args.repo}", flush=True)
    print(f"wall budget: {args.budget_hours:.2f} h", flush=True)
    print(f"search lab: {args.agent_lab}", flush=True)
    print(f"weights: {args.weights}", flush=True)
    print(f"Stockfish: {args.stockfish}", flush=True)
    print(flush=True)

    agent = load_lab_agent(args.agent_lab, args.weights)

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    configure_stockfish(engine, args.sf_hash)

    cache_path = args.output_dir / "stockfish_cache.json"
    cache = load_json(cache_path, {})

    pool = []
    discovery = []
    pathology_cases = []
    deep_rows = []
    static_rows = []
    attributions = []
    timed_rows = []
    stopped_early = False

    try:
        # Position pool.
        pgns = find_pgns(args.repo)
        print(f"PGNs found: {len(pgns)}", flush=True)
        for p in pgns[:20]:
            print(f"  {p}", flush=True)
        if len(pgns) > 20:
            print(f"  ... {len(pgns)-20} more", flush=True)

        raw_cases = []
        raw_cases.extend(positions_from_regression_csv(args.regression_csv))
        raw_cases.extend(positions_from_fen_file(args.fen_suite))
        raw_cases.extend(
            positions_from_pgns(pgns, args.max_pgn_positions, rng)
        )
        raw_cases.extend(
            validation_phase_samples(args.validation, args.validation_per_phase, rng)
        )
        pool = dedup_cases(raw_cases, args.max_pool, rng)
        rng.shuffle(pool)

        write_csv(args.output_dir / "00_position_pool.csv", [asdict(c) for c in pool])
        print(f"Unique position pool: {len(pool)}", flush=True)

        # Stage 1: baseline tail discovery.
        discovery = discovery_scan(
            pool,
            agent,
            engine,
            cache,
            cache_path,
            args.output_dir,
            budget,
            args.screen_agent_depth,
            args.screen_sf_depth,
            args.screen_agent_cap_s,
        )

        screen = select_screen_candidates(discovery, args.screen_candidates)
        write_csv(args.output_dir / "01b_screen_candidates.csv", screen)

        # Stage 2: confirm at actual competition time control.
        pathology_cases = confirm_timed_pathologies(
            screen,
            agent,
            engine,
            cache,
            cache_path,
            args.output_dir,
            budget,
            args.time_left_ms,
            args.confirm_sf_depth,
            args.max_pathologies,
        )

        # Stage 3: deep mechanism ablation.
        deep_rows = deep_ablation(
            pathology_cases,
            agent,
            engine,
            cache,
            cache_path,
            args.output_dir,
            budget,
            args.deep_sf_depth,
            args.deep_agent_depth,
            args.deep_search_cap_s,
        )

        # Stage 4: distinguish static evaluator vs qsearch vs selective search.
        static_rows = static_q_diagnostics(
            pathology_cases, deep_rows, agent, args.output_dir, budget
        )

        attributions = attribute_cases(
            pathology_cases,
            deep_rows,
            static_rows,
            primary_depth=min(5, args.deep_agent_depth),
        )
        write_csv(args.output_dir / "06_attribution.csv", attributions)

        # Stage 5: prove the proposed rescue survives the actual tournament
        # time manager, rather than only working at equal nominal depth.
        timed_rows = timed_attribution_retest(
            pathology_cases,
            attributions,
            agent,
            engine,
            cache,
            cache_path,
            args.output_dir,
            budget,
            args.deep_sf_depth,
            args.time_left_ms,
            args.timed_retest_cases,
        )

    except BudgetExpired:
        stopped_early = True
        print("\nWall-time reserve reached. Writing a complete partial report.", flush=True)
    finally:
        try:
            engine.quit()
        except Exception:
            pass

        # Attribution can still be generated from a partial deep run.
        if deep_rows and not attributions:
            try:
                if not static_rows:
                    # Do not spend remaining time here after the hard budget; just
                    # attribute from whatever mechanism data exists.
                    static_rows = []
                attributions = attribute_cases(
                    pathology_cases,
                    deep_rows,
                    static_rows,
                    primary_depth=min(5, args.deep_agent_depth),
                )
                write_csv(args.output_dir / "06_attribution.csv", attributions)
            except Exception as exc:
                print(f"warning: final attribution failed: {exc!r}", flush=True)

        make_report(
            args.output_dir,
            pool,
            discovery,
            pathology_cases,
            deep_rows,
            attributions,
            timed_rows,
            budget,
            args,
            stopped_early=stopped_early,
        )

    print("\nDONE", flush=True)
    print(f"Report: {args.output_dir / 'FINAL_REPORT.txt'}", flush=True)
    print(f"Attribution: {args.output_dir / '06_attribution.csv'}", flush=True)
    print(f"Pathologies: {args.output_dir / '03_pathology_set.csv'}", flush=True)
    print(f"{budget.status()}", flush=True)


if __name__ == "__main__":
    main()

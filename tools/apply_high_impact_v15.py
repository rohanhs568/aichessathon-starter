#!/usr/bin/env python3
"""Apply the evidence-backed V1.5 search changes to the current V1.4B branch."""

from __future__ import annotations

import py_compile
import shutil
import time
from pathlib import Path


ROOT = Path(".")
AGENT = ROOT / "agent.py"
CANDIDATE_AGENT = ROOT / "candidate" / "agent.py"
ANALYZER = ROOT / "training" / "analyze_pgn_stockfish.py"
INSTALLER = ROOT / "install_and_build_v1_4b_nolmr_rep.sh"
WEIGHTS = ROOT / "weights" / "v1.npz"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"ERROR: {message}")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"ERROR: expected exactly one {label} block, found {count}. "
            "Your local file may differ from commit 38be4ea."
        )
    return text.replace(old, new, 1)


def patch_agent(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    require(
        "V1.4B No-LMR + exact repetition" in text
        or "V1.5 FastKey + wide aspiration + fail-low panic" in text,
        f"{path} is not the expected V1.4B/V1.5 agent",
    )

    if "V1.5 FastKey + wide aspiration + fail-low panic" in text:
        print(f"already patched: {path}")
        return

    text = replace_once(
        text,
        '"""Chessathon learned-evaluation search agent, V1.4B No-LMR + exact repetition.',
        '"""Chessathon learned-evaluation search agent, V1.5 FastKey + wide aspiration + fail-low panic.',
        "version header",
    )

    text = replace_once(
        text,
        "FUTILITY_MARGIN_D1 = 160\n",
        """FUTILITY_MARGIN_D1 = 160

# V1.5 search/time settings. The old 50-unit aspiration window failed far too
# often for the current evaluator's iteration-to-iteration score volatility.
ASPIRATION_WINDOW = 160
ASPIRATION_MAX_WINDOW = 1600

# A completed fail-low is evidence that the previous iteration's PV was too
# optimistic. Give that re-search extra time, but only once per real move and
# never at the expense of the hard clock reserve.
PANIC_TIME_MULTIPLIER = 2.0
PANIC_TIME_MAX_MS = 7_000.0
PANIC_TIME_MIN_CLOCK_MS = 5_000
PANIC_RESERVE_MS = 1_000.0
""",
        "search constants",
    )

    text = replace_once(
        text,
        "NODES = 0\nDEADLINE = 0.0\n",
        """NODES = 0
DEADLINE = 0.0
HARD_DEADLINE = 0.0

ASP_FAIL_LOW_COUNT = 0
ASP_FAIL_HIGH_COUNT = 0
PANIC_EXTENSION_COUNT = 0
""",
        "global timing state",
    )

    text = replace_once(
        text,
        """class SearchTimeout(Exception):
    pass


# ============================================================
# EVALUATION
# ============================================================


def material_evaluate(board):
""",
        """class SearchTimeout(Exception):
    pass


def fast_board_key(board):
    \"\"\"Cheap position identity for caches.

    python-chess already maintains the relevant bitboards incrementally.
    _transposition_key() avoids rebuilding a Polyglot hash at every TT and
    evaluator-cache lookup. Keep a public-API fallback for compatibility.
    \"\"\"
    transposition_key = getattr(board, "_transposition_key", None)
    if transposition_key is not None:
        return transposition_key()
    return chess.polyglot.zobrist_hash(board)


# ============================================================
# EVALUATION
# ============================================================


def material_evaluate(board):
""",
        "fast_board_key insertion",
    )

    text = replace_once(
        text,
        """    zobrist = chess.polyglot.zobrist_hash(board)
    cached = EVAL_CACHE.get(zobrist)
""",
        """    cache_key = fast_board_key(board)
    cached = EVAL_CACHE.get(cache_key)
""",
        "eval cache lookup",
    )

    text = replace_once(
        text,
        """    EVAL_CACHE[zobrist] = score

    return score
""",
        """    EVAL_CACHE[cache_key] = score

    return score
""",
        "eval cache store",
    )

    text = replace_once(
        text,
        """def position_key(board):
    return (
        chess.polyglot.zobrist_hash(board),
        board.halfmove_clock,
    )
""",
        """def position_key(board):
    # Keep the halfmove clock because the 50-move rule makes otherwise
    # identical positions search-distinct. The expensive part was the
    # Polyglot recomputation, not this integer.
    return (
        fast_board_key(board),
        board.halfmove_clock,
    )
""",
        "TT key",
    )

    old_time = """def choose_time_budget(time_left_ms):
    \"\"\"Conservative budget for the 120s + 0.5s tournament clock.\"\"\"
    if time_left_ms <= 250:
        return max(5.0, 0.10 * time_left_ms)

    if time_left_ms <= 1_000:
        return max(15.0, 0.12 * time_left_ms)

    if time_left_ms <= 5_000:
        return min(350.0, 0.055 * time_left_ms + 60.0)

    # Around 3.4s from the initial 120s clock, then naturally less later.
    budget_ms = 0.025 * time_left_ms + 400.0
    budget_ms = min(budget_ms, 3_500.0)

    # Always leave a reserve so a timeout check cannot flag the process.
    budget_ms = min(budget_ms, max(25.0, time_left_ms - 250.0))
    return budget_ms
"""
    new_time = """def choose_time_budget(time_left_ms):
    \"\"\"Conservative normal budget for the 120s + 0.5s tournament clock.\"\"\"
    if time_left_ms <= 250:
        return max(5.0, 0.10 * time_left_ms)

    if time_left_ms <= 1_000:
        return max(15.0, 0.12 * time_left_ms)

    if time_left_ms <= 5_000:
        return min(350.0, 0.055 * time_left_ms + 60.0)

    # Around 3.4s from the initial 120s clock, then naturally less later.
    budget_ms = 0.025 * time_left_ms + 400.0
    budget_ms = min(budget_ms, 3_500.0)

    # Always leave a reserve so a timeout check cannot flag the process.
    budget_ms = min(budget_ms, max(25.0, time_left_ms - 250.0))
    return budget_ms


def choose_panic_budget(time_left_ms, normal_budget_ms):
    \"\"\"Maximum budget available after a completed root fail-low.\"\"\"
    if time_left_ms <= PANIC_TIME_MIN_CLOCK_MS:
        return normal_budget_ms

    safe_clock_budget = max(
        normal_budget_ms,
        time_left_ms - PANIC_RESERVE_MS,
    )
    return min(
        normal_budget_ms * PANIC_TIME_MULTIPLIER,
        PANIC_TIME_MAX_MS,
        safe_clock_budget,
    )


def activate_fail_low_panic():
    \"\"\"Extend this move's deadline to its precomputed hard cap once.\"\"\"
    global DEADLINE
    global PANIC_EXTENSION_COUNT

    if HARD_DEADLINE > DEADLINE:
        DEADLINE = HARD_DEADLINE
        PANIC_EXTENSION_COUNT += 1
"""
    text = replace_once(text, old_time, new_time, "time manager")

    old_asp = """def aspiration_search(board, depth, previous_score):
    if previous_score is None or depth < 4 or abs(previous_score) >= MATE_THRESHOLD:
        return search_root(board, depth, -INF, INF)

    window = 50

    while True:
        alpha = max(-INF, previous_score - window)
        beta = min(INF, previous_score + window)

        move, score = search_root(board, depth, alpha, beta)

        if score <= alpha:
            window *= 2
        elif score >= beta:
            window *= 2
        else:
            return move, score

        if window >= 1600:
            return search_root(board, depth, -INF, INF)
"""
    new_asp = """def aspiration_search(board, depth, previous_score):
    global ASP_FAIL_LOW_COUNT
    global ASP_FAIL_HIGH_COUNT

    if previous_score is None or depth < 4 or abs(previous_score) >= MATE_THRESHOLD:
        return search_root(board, depth, -INF, INF)

    window = ASPIRATION_WINDOW

    while True:
        alpha = max(-INF, previous_score - window)
        beta = min(INF, previous_score + window)

        move, score = search_root(board, depth, alpha, beta)

        if score <= alpha:
            # The previous iteration's PV was materially too optimistic.
            # Buy time for the widened re-search instead of timing out and
            # blindly falling back to that older PV.
            ASP_FAIL_LOW_COUNT += 1
            activate_fail_low_panic()
            window *= 2
        elif score >= beta:
            ASP_FAIL_HIGH_COUNT += 1
            window *= 2
        else:
            return move, score

        if window >= ASPIRATION_MAX_WINDOW:
            return search_root(board, depth, -INF, INF)
"""
    text = replace_once(text, old_asp, new_asp, "aspiration search")

    text = replace_once(
        text,
        """    global NODES
    global DEADLINE
    global _GAME_BOARD
""",
        """    global NODES
    global DEADLINE
    global HARD_DEADLINE
    global ASP_FAIL_LOW_COUNT
    global ASP_FAIL_HIGH_COUNT
    global PANIC_EXTENSION_COUNT
    global _GAME_BOARD
""",
        "get_move globals",
    )

    text = replace_once(
        text,
        """    budget_ms = choose_time_budget(time_left_ms)
    DEADLINE = time.monotonic() + budget_ms / 1000.0
    NODES = 0
""",
        """    budget_ms = choose_time_budget(time_left_ms)
    panic_budget_ms = choose_panic_budget(time_left_ms, budget_ms)

    move_start = time.monotonic()
    DEADLINE = move_start + budget_ms / 1000.0
    HARD_DEADLINE = move_start + panic_budget_ms / 1000.0

    NODES = 0
    ASP_FAIL_LOW_COUNT = 0
    ASP_FAIL_HIGH_COUNT = 0
    PANIC_EXTENSION_COUNT = 0
""",
        "move budget setup",
    )

    old_diag = """        f"total_nodes={NODES} nn={NN_AVAILABLE} "
        f"rep_count={_GAME_COUNTS.get(_rep_key(_GAME_BOARD), 0)}"
"""
    new_diag = """        f"total_nodes={NODES} nn={NN_AVAILABLE} "
        f"rep_count={_GAME_COUNTS.get(_rep_key(_GAME_BOARD), 0)} "
        f"asp_low={ASP_FAIL_LOW_COUNT} asp_high={ASP_FAIL_HIGH_COUNT} "
        f"panic={PANIC_EXTENSION_COUNT}"
"""
    text = replace_once(text, old_diag, new_diag, "final diagnostics")

    path.write_text(text, encoding="utf-8")


def patch_analyzer(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    if "CP_CLAMP = 1_000" in text:
        print(f"already patched: {path}")
        return

    text = replace_once(
        text,
        "MATE_SCORE = 100_000\n",
        """MATE_SCORE = 100_000
CP_CLAMP = 1_000


def clamp_cp(value: int) -> int:
    return max(-CP_CLAMP, min(CP_CLAMP, value))
""",
        "ACPL constants",
    )

    old_loss = """                    best_score = numeric_score(best_info["score"], mover)
                    best_move = best_info.get("pv", [None])[0]

                    board.push(move)

                    played_info = engine.analyse(
                        board,
                        chess.engine.Limit(depth=args.depth),
                    )
                    played_score = numeric_score(played_info["score"], mover)

                    cp_loss = max(0, best_score - played_score)
"""
    new_loss = """                    best_score_raw = numeric_score(best_info["score"], mover)
                    best_score = clamp_cp(best_score_raw)
                    best_move = best_info.get("pv", [None])[0]

                    board.push(move)

                    played_info = engine.analyse(
                        board,
                        chess.engine.Limit(depth=args.depth),
                    )
                    played_score_raw = numeric_score(played_info["score"], mover)
                    played_score = clamp_cp(played_score_raw)

                    # Mate scores are synthetic sentinels, not literal CP.
                    # Clamp each eval before differencing so one mate flip
                    # cannot dominate ACPL.
                    cp_loss = max(0, best_score - played_score)
"""
    text = replace_once(text, old_loss, new_loss, "clamped loss computation")

    old_row = """                        "best_eval_cp": best_score,
                        "played_eval_cp": played_score,
                        "cp_loss": cp_loss,
"""
    new_row = """                        "best_eval_raw_cp": best_score_raw,
                        "played_eval_raw_cp": played_score_raw,
                        "best_eval_cp": best_score,
                        "played_eval_cp": played_score,
                        "cp_loss": cp_loss,
"""
    text = replace_once(text, old_row, new_row, "analyzer output row")

    old_fields = """            "best_eval_cp",
            "played_eval_cp",
            "cp_loss",
"""
    new_fields = """            "best_eval_raw_cp",
            "played_eval_raw_cp",
            "best_eval_cp",
            "played_eval_cp",
            "cp_loss",
"""
    text = replace_once(text, old_fields, new_fields, "analyzer fieldnames")

    text = replace_once(
        text,
        '            f"  ACPL: {acpl:.1f}",\n',
        '            f"  ACPL (evals clamped to +/-{CP_CLAMP}cp): {acpl:.1f}",\n',
        "analyzer summary label",
    )

    path.write_text(text, encoding="utf-8")


def patch_installer(path: Path) -> None:
    if not path.exists():
        return

    text = path.read_text(encoding="utf-8")
    text = text.replace(
        '"${PY[@]}" training/verify_candidate_capture.py',
        '"${PY[@]}" -m training.verify_candidate_capture',
    )
    text = text.replace(
        '"${PY[@]}" training/test_candidate_repetition.py',
        '"${PY[@]}" -m training.test_candidate_repetition',
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    require(AGENT.is_file(), "agent.py not found; run from repo root")
    require(CANDIDATE_AGENT.is_file(), "candidate/agent.py not found")
    require(ANALYZER.is_file(), "training/analyze_pgn_stockfish.py not found")
    require(WEIGHTS.is_file(), "weights/v1.npz not found")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = ROOT / "candidate_backups" / f"pre_v15_{stamp}"
    backup.mkdir(parents=True, exist_ok=True)

    shutil.copy2(AGENT, backup / "agent.py")
    shutil.copy2(CANDIDATE_AGENT, backup / "candidate_agent.py")
    shutil.copy2(ANALYZER, backup / "analyze_pgn_stockfish.py")
    shutil.copy2(WEIGHTS, backup / "v1.npz")

    baseline = ROOT / "baselines" / "v1_4b_pre_fastasp"
    (baseline / "weights").mkdir(parents=True, exist_ok=True)
    shutil.copy2(AGENT, baseline / "agent.py")
    shutil.copy2(WEIGHTS, baseline / "weights" / "v1.npz")

    patch_agent(AGENT)
    patch_agent(CANDIDATE_AGENT)
    patch_analyzer(ANALYZER)
    patch_installer(INSTALLER)

    for path in (AGENT, CANDIDATE_AGENT, ANALYZER):
        py_compile.compile(str(path), doraise=True)

    require(
        AGENT.read_bytes() == CANDIDATE_AGENT.read_bytes(),
        "root agent.py and candidate/agent.py drifted",
    )

    print()
    print("V1.5 HIGH-IMPACT SEARCH CHANGES APPLIED")
    print(f"Backup: {backup}")
    print("Baseline snapshot: baselines/v1_4b_pre_fastasp")
    print()
    print("Next:")
    print("  .venv/bin/python -m training.verify_candidate_capture")
    print("  .venv/bin/python -m training.test_candidate_repetition")
    print("  ./run_v15_high_impact_checks.sh")


if __name__ == "__main__":
    main()

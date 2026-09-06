#!/usr/bin/env python3
"""Patch the current local V1.5 agent to V1.6 robust fail-low handling."""

from __future__ import annotations

import py_compile
import shutil
import time
from pathlib import Path

ROOT = Path(".")
FILES = [ROOT / "agent.py", ROOT / "candidate" / "agent.py"]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"ERROR: {label}: expected 1 match, found {n}")
    return text.replace(old, new, 1)


def patch(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    if "V1.6 FastKey + robust fail-low" in text:
        print(f"already V1.6: {path}")
        return

    if "V1.5 FastKey + wide aspiration + fail-low panic" not in text:
        raise SystemExit(
            f"ERROR: {path} is not the expected local V1.5 agent. "
            "Do not apply this to an unknown search version."
        )

    text = replace_once(
        text,
        "V1.5 FastKey + wide aspiration + fail-low panic",
        "V1.6 FastKey + robust fail-low",
        "version",
    )

    old = """        if score <= alpha:
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

    new = """        if score <= alpha:
            # The previous iteration's PV has now been disproved at this
            # depth. Spend the panic budget and resolve the current depth
            # directly with one full-window root search instead of burning
            # time through repeated low-window aspiration re-searches.
            ASP_FAIL_LOW_COUNT += 1
            activate_fail_low_panic()
            return search_root(board, depth, -INF, INF)

        if score >= beta:
            # Fail-high is less dangerous: the old PV has not collapsed.
            # Keep ordinary exponential widening for this direction.
            ASP_FAIL_HIGH_COUNT += 1
            window *= 2
            if window >= ASPIRATION_MAX_WINDOW:
                return search_root(board, depth, -INF, INF)
            continue

        return move, score
"""

    text = replace_once(text, old, new, "aspiration fail-low block")
    path.write_text(text, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)


def main() -> None:
    for path in FILES:
        if not path.is_file():
            raise SystemExit(f"ERROR: missing {path}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = ROOT / "candidate_backups" / f"pre_v16_{stamp}"
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FILES[0], backup / "agent.py")
    shutil.copy2(FILES[1], backup / "candidate_agent.py")

    for path in FILES:
        patch(path)

    if FILES[0].read_bytes() != FILES[1].read_bytes():
        raise SystemExit("ERROR: root and candidate agents drifted")

    print()
    print("V1.6 robust fail-low handling installed.")
    print(f"Backup: {backup}")
    print("Run:")
    print("  .venv/bin/python -m training.verify_candidate_capture")
    print("  .venv/bin/python -m training.test_candidate_repetition")


if __name__ == "__main__":
    main()

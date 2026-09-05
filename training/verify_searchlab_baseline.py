"""Fixed-depth equivalence check: tournament agent.py vs searchlab baseline.

This avoids wall-clock noise. At each FEN both engines search iterative depths
1..D with effectively infinite deadlines. Move, score and node count must match
at every completed depth. If they do not, do NOT trust the ablation results.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import os
import time
from contextlib import redirect_stdout
from pathlib import Path

import chess


def load_module(name, path, weights=None):
    if weights is not None:
        os.environ["SEARCHLAB_MODEL_PATH"] = str(weights.resolve())
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def reset_original(agent):
    agent.TT.clear()
    agent.EVAL_CACHE.clear()
    agent.KILLERS.clear()
    agent.HISTORY.clear()
    agent.NODES = 0
    agent.DEADLINE = time.monotonic() + 3600


def reset_lab(agent):
    agent.set_searchlab_variant("baseline")
    agent.reset_searchlab_state(reset_game=True)
    agent.NODES = 0
    agent.DEADLINE = time.monotonic() + 3600
    agent._GAME_COUNTS = {}
    agent._PATH_COUNTS = {}


def load_fens(path, max_fens):
    fens = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            chess.Board(line)
            fens.append(line)
            if max_fens and len(fens) >= max_fens:
                break
    return fens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, default=Path("agent.py"))
    parser.add_argument("--lab", type=Path, default=Path("training/searchlab_agent.py"))
    parser.add_argument("--weights", type=Path, default=Path("weights/v1.npz"))
    parser.add_argument("--fens", type=Path, default=Path("training/data/test_fens_60.txt"))
    parser.add_argument("--max-fens", type=int, default=10)
    parser.add_argument("--depth", type=int, default=5)
    args = parser.parse_args()

    original = load_module("original_agent_equiv", args.original)
    lab = load_module("searchlab_agent_equiv", args.lab, args.weights)
    fens = load_fens(args.fens, args.max_fens)

    failures = []
    for i, fen in enumerate(fens, start=1):
        reset_original(original)
        reset_lab(lab)
        board_a = chess.Board(fen)
        board_b = chess.Board(fen)

        for depth in range(1, args.depth + 1):
            before_a = original.NODES
            move_a, score_a = original.search_root(board_a, depth, -original.INF, original.INF)
            nodes_a = original.NODES - before_a

            before_b = lab.NODES
            move_b, score_b = lab.search_root(board_b, depth, -lab.INF, lab.INF)
            nodes_b = lab.NODES - before_b

            ua = None if move_a is None else move_a.uci()
            ub = None if move_b is None else move_b.uci()
            ok = ua == ub and score_a == score_b and nodes_a == nodes_b
            print(
                f"fen={i:02d} depth={depth} "
                f"original={ua}/{score_a}/{nodes_a} "
                f"lab={ub}/{score_b}/{nodes_b} {'PASS' if ok else 'FAIL'}"
            )
            if not ok:
                failures.append((i, depth, ua, score_a, nodes_a, ub, score_b, nodes_b))

    if failures:
        print("\nBASELINE EQUIVALENCE FAILED")
        for item in failures[:20]:
            print(item)
        raise SystemExit(2)

    print(f"\nBASELINE EQUIVALENCE PASS: {len(fens)} FENs through depth {args.depth}")


if __name__ == "__main__":
    main()

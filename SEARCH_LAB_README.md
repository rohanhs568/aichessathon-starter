# Chessathon V1 Search Lab

This bundle is designed to answer **which search feature is actually causing the catastrophic moves**, without changing evaluator weights at the same time.

It is deliberately an experimental sidecar. It does **not** replace `agent.py`.

## What is tested

The search-lab baseline reproduces V1 FastQ, then the runner performs paired ablations on identical FENs:

| Variant | Change |
|---|---|
| `baseline` | Current V1 FastQ semantics |
| `no_null` | Null-move pruning disabled |
| `null_safe` | Null allowed only from depth >= 4 and never directly to qsearch |
| `null_no_endgame` | Current null, but disabled at <=10 total pieces |
| `no_lmr` | LMR disabled |
| `lmr_safe` | LMR starts later and is capped at one ply |
| `no_futility` | Frontier futility disabled |
| `futility_wide` | D1 futility margin 160 -> 300 cp |
| `no_selectivity` | Null + LMR + futility all disabled |
| `capture_material_order` | Does not prune captures; superficially losing captures are only moved below killers |
| `no_aspiration` | Full root window at every iteration |
| `no_pvs` | Full-window later moves; expensive diagnostic only |
| `conservative` | Safer null + safer LMR + wider futility + capture ordering together |

The lab agent also has `repetition_exact` and `repetition_cycle`, but repetition is tested separately because an isolated FEN does not contain its game history.

## Why these tests

Our current search is highly selective relative to its completed depth. At interior nodes it combines null-move pruning, LMR, D1 futility, PVS, TT cutoffs, and capture-first ordering. Any one of those can be useful while their interaction can still hide a critical quiet refutation.

The experiment is **paired**: every variant sees the same position, same evaluator weights, same nominal tournament clock input, and the same Stockfish reference settings.

## The scoring logic

This bundle intentionally does **not** rank variants only by average centipawn loss.

For every candidate move Stockfish is run on the **original root board** with:

```python
engine.analyse(board, limit, root_moves=[candidate])
```

and all scores are converted to the **root player's point of view**. This avoids the classic sign error caused by pushing the move and forgetting that `board.turn` has flipped.

Two metrics are reported:

1. **WDL expectation loss**: primary metric. Bounded and meaningful for game outcome.
2. **CP loss**: secondary diagnostic. Intuitive near equality and useful for blunder thresholds.

CP loss is capped at 1000 only for the *mean* summary so mate scores do not dominate. Raw CP loss, mate blunders, sign flips, and win-to-nonwin events are still reported separately.

The suite also records search internals such as null tries/cutoffs, LMR reductions/re-searches, futility skips, TT cutoffs and completed depth.

## Before trusting any ablation

Run the fixed-depth baseline equivalence test:

```bash
uv run python -m training.verify_searchlab_baseline \
  --original agent.py \
  --lab training/searchlab_agent.py \
  --fens training/data/test_fens_60.txt \
  --max-fens 10 \
  --depth 5
```

It requires **move, score and node count to match** at every depth. This removes wall-clock noise from the verification. If this fails, do not use the ablation results until the lab copy is corrected.

## CP/tanh audit first

Run:

```bash
uv run python -m training.cp_tanh_diagnostics \
  --checkpoint training/models/v1_natural_600m/final.pt \
  --validation training/data/samples/lichess_validation_250k.csv \
  --output-dir cp_tanh_diagnostics
```

Or use the 1.4b checkpoint:

```bash
uv run python -m training.cp_tanh_diagnostics \
  --checkpoint training/models/v1_sf_long_1800m/checkpoint_1.4b.pt \
  --validation training/data/samples/lichess_validation_250k.csv \
  --output-dir cp_tanh_diagnostics_1.4b
```

This computes the current runtime-equivalent CP prediction as:

```text
raw_output * K
```

and compares it against the old validation calculation:

```text
K * atanh(clip(tanh(raw_output), -0.999, 0.999))
```

The two are mathematically identical **before clipping**. At K=400 the legacy `0.999` clip corresponds to about 1520 cp, so predictions beyond that can distort the reported overall CP MAE. Balanced and medium metrics should be much less affected, but the script measures the actual effect rather than assuming it.

It also writes target-saturation diagnostics for K=400, 600, 800 and 1000. Do not compare transformed target MAE across different K values; compare raw-CP metrics and chess move quality.

## Repetition design test

```bash
uv run python -m training.test_repetition_design
```

It verifies:

- python-chess threefold semantics,
- that the FEN-only tournament protocol contains enough information to reconstruct the one opponent move between calls,
- exact third-occurrence scoring,
- optional first-cycle search-tree scoring.

`exact` and `cycle` are intentionally different policies. Exact threefold handling is correctness. First-cycle draw scoring is a search heuristic and should be A/B tested.

## First search run: fixed depth, then tournament clock

Run the same variants at a fixed nominal depth first:

```bash
uv run python -m training.search_ablation_suite \
  --fens training/data/test_fens_60.txt \
  --max-fens 20 \
  --preset core \
  --mode fixed-depth \
  --agent-depth 5 \
  --stockfish /usr/games/stockfish \
  --sf-depth 18 \
  --output-dir search_lab_results_core20_fixed
```

This isolates **search correctness/selectivity at the same nominal depth**. If `no_lmr` rescues a blunder here, for example, that is direct evidence that the reduction is hiding the refutation rather than merely changing how much depth fits into the clock.

Then run the tournament-clock comparison:

```bash
uv run python -m training.search_ablation_suite \
  --fens training/data/test_fens_60.txt \
  --max-fens 20 \
  --preset core \
  --mode timed \
  --time-left-ms 120000 \
  --stockfish /usr/games/stockfish \
  --sf-depth 18 \
  --output-dir search_lab_results_core20_timed
```

The V1 time manager turns 120000 ms remaining into about 3.4 s per move. The timed run measures the real trade-off: safer search may examine more of each branch but complete less nominal depth. Stockfish evaluations are cached by FEN and candidate move, so later reruns are cheaper.

Read:

```text
search_lab_results_core20/search_ablation_summary.txt
search_lab_results_core20/search_ablation_summary.csv
search_lab_results_core20/search_ablation_pairwise.csv
search_lab_results_core20/search_ablation_raw.csv
```

## Then run all 60 positions

Only after the 20-position run looks sane:

```bash
uv run python -m training.search_ablation_suite \
  --fens training/data/test_fens_60.txt \
  --preset core \
  --time-left-ms 120000 \
  --stockfish /usr/games/stockfish \
  --sf-depth 18 \
  --output-dir search_lab_results_core60
```

## Build a regression suite from actual bad games

This is more important than generic FENs for the sacrifice problem.

Save/export PGNs from the games where the bot made bizarre captures, sacrifices, or failed to convert. Then:

```bash
uv run python -m training.extract_blunder_cases games/*.pgn \
  --stockfish /usr/games/stockfish \
  --depth 18 \
  --min-cp-loss 50 \
  --min-expectation-loss 0.02 \
  --out training/data/search_regression_cases.csv
```

Run the same ablation on those positions:

```bash
uv run python -m training.search_ablation_suite \
  --cases-csv training/data/search_regression_cases.csv \
  --preset core \
  --time-left-ms 120000 \
  --stockfish /usr/games/stockfish \
  --sf-depth 20 \
  --output-dir search_lab_real_blunders
```

## Wide second pass

After the primary suspects are understood:

```bash
uv run python -m training.search_ablation_suite \
  --cases-csv training/data/search_regression_cases.csv \
  --preset wide \
  --time-left-ms 120000 \
  --sf-depth 20 \
  --output-dir search_lab_wide
```

`no_pvs` can be substantially slower. It is a diagnostic, not an expected production setting.

## How to decide what to keep

Do not choose the smallest average CP loss blindly. A promising change should normally satisfy most of these:

- lower paired **WDL expectation loss** than baseline,
- fewer `>=150 cp` and `>=300 cp` failures,
- rescues real website blunders without introducing comparable new ones,
- does not materially increase mate blunders or win-to-nonwin transitions,
- acceptable completed depth / node cost,
- the paired bootstrap interval is at least trending consistently rather than one spectacular FEN dominating the mean,
- then confirms in self-play with the evaluator held fixed.

Interpretation examples:

- `no_null` much better but much shallower -> try `null_safe` or `null_no_endgame` rather than deleting null outright.
- `no_lmr` rescues blunders but loses depth -> prefer `lmr_safe`, then tune start depth/move number.
- only `no_selectivity` works -> likely interaction between pruning mechanisms rather than one isolated bug.
- `capture_material_order` improves without changing node count much -> the issue is more likely move ordering / alpha poisoning than pruning correctness.
- `no_aspiration` makes no quality difference -> keep aspiration because it is probably not the problem.

## Relevant references

- Chess Programming Wiki, Null Move Pruning: https://chessprogramming.org/Null_Move_Pruning
- Chess Programming Wiki, Late Move Reductions: https://chessprogramming.org/Late_Move_Reductions
- Chess Programming Wiki, Quiescence Search: https://chessprogramming.org/Quiescence_Search
- Chess Programming Wiki, Repetitions: https://chessprogramming.org/Repetitions
- Chess Programming Wiki, Static Exchange Evaluation: https://chessprogramming.org/Static_Exchange_Evaluation
- Current Stockfish search implementation: https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp
- python-chess engine score/WDL API: https://python-chess.readthedocs.io/en/latest/engine.html
- Stockfish NNUE documentation: https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md


## Repetition-test correction (2026-09-05)

The first bundle's deterministic protocol test incorrectly required the
reconstructed `Board.move_stack` length to equal the synthetic reference board's
stack length. The production protocol supplies only FENs, so exact stack length
is not itself the invariant we need. The corrected design stores explicit
real-game repetition counts across calls and reconstructs only the single
opponent move needed to advance those counts.

Use `run_search_lab_from_repetition.sh` after replacing the corrected files to
resume at section 2 without repeating the already-passed tanh audit and baseline
equivalence check.

# Chessathon Deep Pathology Diagnostic

This is an unattended diagnostic pipeline intended to run while you are away.
It does not modify `agent.py` or `weights/v1.npz`.

## What it does

1. Builds a broad pool of real positions from repository PGNs, the existing
   60-FEN test suite, previous regression cases, and a phase-stratified sample
   of the 250k permanent Stockfish validation set.
2. Searches every position with the current V1 FastQ **baseline** at fixed
   depth 5 and finds the tail where Stockfish says the selected move loses
   meaningful WDL expectation / CP.
3. Rechecks the strongest tail positions using the **actual tournament time
   manager**. Only these confirmed failures become the pathology set.
4. On each pathology, runs a depth ladder with individual and pairwise search
   ablations. This distinguishes a single bad mechanism from interactions.
5. Evaluates every unique move produced at every depth using deeper Stockfish,
   always from the original root position.
6. Compares NN one-ply static scores and current qsearch scores for the
   baseline move versus the Stockfish move.
7. Produces an automatic causal attribution and then retests the proposed
   rescue at the actual tournament clock.

## Main output files

- `00_position_pool.csv`: all positions considered.
- `01_discovery_all.csv`: baseline tail scan.
- `02_timed_confirmation_all.csv`: tournament-clock confirmation.
- `03_pathology_set.csv`: the final pathological/tail positions.
- `03_pathology_fens.txt`: same set as simple FENs.
- `04_deep_ablation_raw.csv`: every variant, every completed depth, Stockfish loss.
- `05_static_qsearch_diagnostics.csv`: NN static vs qsearch comparison.
- `06_attribution.csv`: mechanism attribution per case.
- `07_timed_attribution_retest.csv`: whether the proposed fix survives real timing.
- `FINAL_REPORT.txt`: human-readable summary.

## Attribution meanings

- `single:no_lmr`, `single:no_null`, etc.: removing that mechanism at the SAME
  nominal depth materially improves the move.
- `interaction:no_null_no_lmr`, etc.: neither single removal was enough, but
  the pair was.
- `selectivity:higher_order`: only broad selectivity removal rescued the case.
- `depth_horizon`: the ordinary baseline solves it when searched deeper.
- `evaluator_static_suspect`: the NN one-ply score itself strongly prefers the
  bad move over Stockfish's best move.
- `qsearch_horizon_suspect`: the static NN does not strongly prefer the bad
  move, but the current qsearch does.
- `unresolved_mixed_or_depth`: none of the tested mechanisms isolates it.

## Metrics

The primary metric is Stockfish WDL expectation loss from the root player's
point of view. CP loss is secondary. Every candidate is analysed on the
ORIGINAL root using `root_moves=[candidate]`, which prevents a side-to-move sign
error.

## Start it

Place `deep_pathology_pipeline.py` in `training/` and
`start_deep_pathology_5h.sh` in the repository root. Then:

```bash
chmod +x start_deep_pathology_5h.sh
./start_deep_pathology_5h.sh
```

It launches under `nohup` and returns the shell immediately.

Check it with:

```bash
tail -f deep_pathology_master.log
```

`Ctrl+C` stops only `tail`, not the background experiment.

The pipeline has a 4.5-hour wall budget and keeps incremental CSVs. If it hits
the wall-time reserve, it writes a partial `FINAL_REPORT.txt` rather than losing
work.

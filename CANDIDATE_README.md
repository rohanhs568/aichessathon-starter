# Chessathon candidate: 1.4b evaluator + No-LMR + exact threefold

This is the evidence-based intermediate competition build.

Search changes relative to V1 FastQ:
- LMR disabled.
- Exact threefold occurrence tracking added across persistent tournament calls.
- Repetition checked before TT use and scored as 0.
- TT cleared at the start of each real move, but retained across iterative-deepening
  iterations inside that move.
- Null-move subtrees do not inherit real-game repetition semantics.
- Null move, depth-1 futility, PVS, aspiration windows, FastQ, move ordering,
  time management, and evaluator architecture otherwise remain unchanged.

Evaluator:
- The installer exports the saved ~1.4b Stockfish continuation checkpoint.
- It refuses to deploy a checkpoint with positions_seen < 1.35b.

Safety gates run automatically before packaging:
1. PyTorch -> exported NPZ -> runtime inference equivalence.
2. Legal-move smoke tests.
3. Simple hanging-material capture tests at depths 1 and 3.
4. Exact FEN-only threefold reconstruction test.
5. Short harness game vs random.
6. Tournament package build.

From repo root:

    unzip -o chessathon_v1_4b_nolmr_rep_deploy.zip
    chmod +x install_and_build_v1_4b_nolmr_rep.sh rollback_candidate.sh
    ./install_and_build_v1_4b_nolmr_rep.sh

If all checks pass, upload:

    submission_v1_4b_nolmr_rep.zip

Rollback:

    ./rollback_candidate.sh

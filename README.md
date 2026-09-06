# V1.6 blunder fix

This bundle addresses the two mechanisms that actually showed up in profiling.

## 1. Robust fail-low handling

The local V1.5 already has:
- fast cache keys;
- aspiration widened to 160;
- panic time on fail-low.

V1.6 makes the fail-low path more decisive:

```text
completed fail-low
    -> old PV has been disproved at this depth
    -> activate panic time
    -> immediately re-search this depth with a full root window
```

It no longer burns time on 160 -> 320 -> 640 -> ... low-window retries.

Apply:

```bash
.venv/bin/python tools/apply_v16_blunder_fix.py
.venv/bin/python -m training.verify_candidate_capture
.venv/bin/python -m training.test_candidate_repetition
```

## 2. Fix the evaluator objective that creates arbitrary material sacrifices

The current V1 parameters were trained with:

```text
prediction = tanh(raw)
target     = tanh(cp / K)
MSE loss
```

but tournament search uses:

```text
score = raw * K
```

V1.6 keeps the exact same 50k-parameter network and starts from the 1.4b
checkpoint, but directly trains the runtime quantity:

```text
prediction = raw
target     = clip(cp, -2000, 2000) / 400
loss       = Huber(beta = 200cp)
```

This removes the final tanh gradient saturation from recalibration. The tensor
names and architecture are unchanged, so the existing NPZ exporter and runtime
remain compatible.

Start a 100m-position recalibration:

```bash
chmod +x start_v16_recalibration.sh \
  inspect_v16_recalibration.sh \
  install_linear_checkpoint.sh

./start_v16_recalibration.sh
```

Watch:

```bash
tail -f v1_linear_huber_100m.log
```

Checkpoints are written at 25m / 50m / 75m / 100m.

Compare the legacy 1.4b network to every available linear checkpoint:

```bash
./inspect_v16_recalibration.sh
```

The diagnostic prints:
- clipped CP MAE;
- balanced and medium CP MAE;
- sign accuracy;
- calibration slope on |teacher| <= 1000cp;
- correlation;
- residual standard deviation;
- median implied pawn/knight/bishop/rook/queen values;
- rook/bishop ratio.

The important gates are not just MAE. We want:
- calibration slope materially closer to 1;
- lower residual noise;
- rook/bishop no longer near ~1.1;
- no meaningful collapse in sign accuracy.

When one checkpoint is clearly better, install it:

```bash
./install_linear_checkpoint.sh \
  training/models/v1_linear_huber_100m/checkpoint_linear_50m.pt
```

That script exports to a temporary NPZ and verifies the raw PyTorch output
against NPZ runtime arithmetic before replacing `weights/v1.npz`.

## Not mixed into this change

No SEE, check extension, material blend, LMR, root hysteresis, or incomplete
root-iteration commit. Those remain separate after the evaluator is calibrated.

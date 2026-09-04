#!/usr/bin/env bash
set -euo pipefail

GPU_PY="${GPU_PY:-.venv-gpu/bin/python}"
SHARDS="${SHARDS:-/mnt/d/ChessData/lichess_train_shards}"
VALIDATION="${VALIDATION:-training/data/samples/lichess_validation_250k.csv}"
RUN_DIR="${RUN_DIR:-/tmp/v1_gpu_preflight}"
LOG="${LOG:-/tmp/v1_gpu_preflight.log}"

if [[ ! -x "$GPU_PY" ]]; then
  echo "GPU Python not found: $GPU_PY"
  exit 2
fi

rm -rf "$RUN_DIR"
rm -f "$LOG"

echo "=== CUDA capability check ==="

"$GPU_PY" - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")

print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("wheel arch list:", torch.cuda.get_arch_list())

major, minor = torch.cuda.get_device_capability(0)
wanted = f"sm_{major}{minor}"

if wanted not in torch.cuda.get_arch_list():
    raise SystemExit(
        f"PyTorch wheel does not contain native support for {wanted}"
    )

# Basic allocation/kernel/synchronization loop.
x = torch.randn(4096, 256, device="cuda")
w = torch.randn(256, 64, device="cuda", requires_grad=True)

for i in range(200):
    y = (x @ w).square().mean()
    y.backward()
    with torch.no_grad():
        w -= 1e-5 * w.grad
        w.grad = None
    if i % 20 == 0:
        torch.cuda.synchronize()

torch.cuda.synchronize()
print("basic CUDA stress: PASS")
PY

echo
echo "=== Real V1 training preflight: 20m positions ==="
echo "Log: $LOG"

set +e
CUDA_LAUNCH_BLOCKING=1 \
"$GPU_PY" -m training.train_v1 \
  --shards "$SHARDS" \
  --validation "$VALIDATION" \
  --run-dir "$RUN_DIR" \
  --hidden 64 \
  --buckets 8 \
  --batch-size 16384 \
  --max-positions 20m \
  --checkpoints 20m \
  --log-every 1m \
  >"$LOG" 2>&1
status=$?
set -e

cat "$LOG"

if [[ $status -ne 0 ]]; then
  echo
  echo "GPU V1 preflight: FAIL (exit $status)"
  exit "$status"
fi

if ! grep -q "Final validation" "$LOG" && ! grep -q "Saved final model" "$LOG"; then
  # train_v1 versions differ slightly in the final message. A final.pt is the
  # definitive success condition.
  if [[ ! -f "$RUN_DIR/final.pt" ]]; then
    echo "GPU V1 preflight did not create final.pt"
    exit 3
  fi
fi

echo
echo "GPU V1 preflight: PASS"
echo "Recent throughput lines:"
grep 'rate=' "$LOG" | tail -n 5 || true

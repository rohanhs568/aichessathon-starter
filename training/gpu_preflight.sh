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
import re
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")

name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
arch_list = torch.cuda.get_arch_list()

print("GPU:", name)
print("capability:", (major, minor))
print("wheel arch list:", arch_list)

# CUDA cubins are binary-compatible within the same compute-capability major
# version when the GPU minor version is >= the cubin minor version.
# Therefore a 6.1 GPU can run an sm_60 cubin.
compatible_native = []

for arch in arch_list:
    match = re.fullmatch(r"sm_(\d)(\d)", arch)
    if not match:
        continue

    arch_major = int(match.group(1))
    arch_minor = int(match.group(2))

    if arch_major == major and arch_minor <= minor:
        compatible_native.append((arch_major, arch_minor, arch))

if not compatible_native:
    raise SystemExit(
        f"No compatible native cubin found for compute capability "
        f"{major}.{minor}"
    )

best = max(compatible_native)
print("compatible native cubin:", best[2])

# Basic real CUDA allocation/kernel/backprop/synchronization stress.
x = torch.randn(4096, 256, device="cuda")
w = torch.randn(256, 64, device="cuda", requires_grad=True)

for i in range(300):
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

if [[ ! -f "$RUN_DIR/final.pt" ]]; then
  echo
  echo "GPU V1 preflight did not create $RUN_DIR/final.pt"
  exit 3
fi

echo
echo "GPU V1 preflight: PASS"
echo "Recent throughput lines:"
grep 'rate=' "$LOG" | tail -n 5 || true

#!/usr/bin/env bash
# The only reduced launch workflow: same registered model, CUDA/NCCL/SyncBN and data path.
set -euo pipefail
slot=${1:-2}
if (( $# > 0 )); then
    shift
fi
: "${ACCEPTANCE_DIR:?Set ACCEPTANCE_DIR to a dedicated acceptance output directory}"

bash scripts/run_4gpu.sh "$slot" --mode acceptance --acceptance-root "$ACCEPTANCE_DIR" --stop-after-epoch 1 "$@"
bash scripts/run_4gpu.sh "$slot" --mode acceptance --acceptance-root "$ACCEPTANCE_DIR" --resume

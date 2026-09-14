#!/usr/bin/env bash
set -euo pipefail

slot=${1:?Usage: bash scripts/run_4gpu.sh SLOT [--resume | --retry-initial]}
shift
: "${DATA_DIR:?Set DATA_DIR to the prepared dataset directory}"
: "${CAMPAIGN_DIR:?Set CAMPAIGN_DIR to the campaign directory}"
: "${ENV_PROFILE:?Set ENV_PROFILE to the recorded four-GPU profile}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
log_root="${CAMPAIGN_DIR}/launcher_logs"
mkdir -p "$log_root"
launch_dir=$(mktemp -d "${log_root}/slot_${slot}_XXXXXX")
set +e
"${PYTHON:-python}" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
    --log-dir "$launch_dir" --tee 3 --module vgae_cf train \
    --campaign "$CAMPAIGN_DIR" --slot "$slot" --data "$DATA_DIR" \
    --environment "$ENV_PROFILE" --workers "${DATA_WORKERS:-0}" "$@" \
    2>&1 | tee "$launch_dir/launcher.log"
exit_code=${PIPESTATUS[0]}
set -e
printf '%s\n' "$exit_code" > "$launch_dir/exit_code.txt"
exit "$exit_code"

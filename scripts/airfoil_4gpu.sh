#!/usr/bin/env bash
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
: "${DATA_DIR:?Set DATA_DIR to the prepared Airfoil directory}"
: "${RESULT_ROOT:?Set a new RESULT_ROOT for this training run}"
: "${CUDA_VISIBLE_DEVICES:?Set four allocated GPU IDs or use the scheduler mask}"
IFS=',' read -r -a assigned <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#assigned[@]} != 4 )); then printf 'Exactly four GPUs are required.\n' >&2; exit 2; fi
python_bin=${PYTHON:-python}
action=${1:-train}
case "$action" in
    prepare) exec "$python_bin" -m vgae_cf prepare --data "$DATA_DIR" ;;
    train|resume) ;;
    *) printf 'Usage: bash scripts/airfoil_4gpu.sh prepare|train|resume\n' >&2; exit 2 ;;
esac
if [[ "$action" == train && -e "$RESULT_ROOT" ]]; then
    printf 'Choose a new RESULT_ROOT or use resume.\n' >&2; exit 2
fi
mkdir -p "$RESULT_ROOT"
export CAMPAIGN_DIR="$RESULT_ROOT/campaign"
export ENV_PROFILE="$RESULT_ROOT/environment.json"
"$python_bin" -m vgae_cf campaign create --root "$CAMPAIGN_DIR"
"$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4     --module vgae_cf environment --output "$ENV_PROFILE"
extra=()
if [[ "$action" == resume ]]; then extra=(--resume); fi
log_dir=$(mktemp -d "$RESULT_ROOT/launcher_XXXXXX")
set +e
"$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4     --log-dir "$log_dir" --tee 3 --module vgae_cf train     --campaign "$CAMPAIGN_DIR" --slot 0 --data "$DATA_DIR" --environment "$ENV_PROFILE"     --workers "${DATA_WORKERS:-0}" "${extra[@]}" 2>&1 | tee "$log_dir/launcher.log"
rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$rc" > "$log_dir/exit_code.txt"
exit "$rc"

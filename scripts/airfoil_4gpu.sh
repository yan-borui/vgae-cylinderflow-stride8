#!/usr/bin/env bash
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
python_bin=${PYTHON:-python}
action=${1:-train}
case "$action" in
    data|prepare|train|resume) ;;
    *) printf 'Usage: bash scripts/airfoil_4gpu.sh data|prepare|train|resume\n' >&2; exit 2 ;;
esac
export RAW_DATA_DIR=${RAW_DATA_DIR:-/data/datasets/meshgraphnets/airfoil}
export DATA_DIR=${DATA_DIR:-${RAW_DATA_DIR}_uvp_stride8}
if [[ "$action" == train || "$action" == resume ]]; then
    : "${RESULT_ROOT:?Set RESULT_ROOT for this training run}"
    : "${CUDA_VISIBLE_DEVICES:?Set four allocated GPU IDs or use the scheduler mask}"
    IFS=',' read -r -a assigned <<< "$CUDA_VISIBLE_DEVICES"
    if (( ${#assigned[@]} != 4 )); then printf 'Exactly four GPUs are required.\n' >&2; exit 2; fi
    if [[ "$action" == train && -e "$RESULT_ROOT" ]]; then
        printf 'Choose a new RESULT_ROOT or use resume.\n' >&2; exit 2
    fi
fi
data_options=()
if [[ "${AIRFOIL_OFFLINE:-0}" == 1 ]]; then data_options=(--offline); fi
"$python_bin" -m airfoil_data.ensure --raw-dir "$RAW_DATA_DIR" --output-dir "$DATA_DIR" "${data_options[@]}"
if [[ "$action" == data ]]; then exit 0; fi
data="$DATA_DIR/airfoil_stride8_75frames.h5"
manifest="$DATA_DIR/airfoil_stride8_75frames_manifest.json"
extra=()
if [[ "$action" == resume ]]; then extra=(--resume); fi

(
    flock 9
    if [[ ! -f "$DATA_DIR/graph_inventory.json" ]]; then
        "$python_bin" -m vgae_cf prepare --data "$DATA_DIR" 2>&1 | tee -a "$DATA_DIR/vgae_graph_preparation.log"
    fi
) 9>"$DATA_DIR/.vgae-graphs.lock"
if [[ "$action" == prepare ]]; then exit 0; fi
mkdir -p "$RESULT_ROOT"
export CAMPAIGN_DIR="$RESULT_ROOT/campaign"
export ENV_PROFILE="$RESULT_ROOT/environment.json"
"$python_bin" -m vgae_cf campaign create --root "$CAMPAIGN_DIR"
"$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
    --module vgae_cf environment --output "$ENV_PROFILE"
log_dir=$(mktemp -d "$RESULT_ROOT/launcher_XXXXXX")
set +e
"$python_bin" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
    --log-dir "$log_dir" --tee 3 --module vgae_cf train \
    --campaign "$CAMPAIGN_DIR" --slot 0 --data "$DATA_DIR" --environment "$ENV_PROFILE" \
    --workers "${DATA_WORKERS:-0}" "${extra[@]}" 2>&1 | tee "$log_dir/launcher.log"
rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$rc" > "$log_dir/exit_code.txt"
exit "$rc"

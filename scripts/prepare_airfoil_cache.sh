#!/usr/bin/env bash
set -euo pipefail
code_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$code_root"
python_bin=${PYTHON:-python}
data="$DATA_DIR/airfoil_stride8_75frames.h5"
manifest="$DATA_DIR/airfoil_stride8_75frames_manifest.json"
if [[ ! -f "$DATA_DIR/graph_inventory.json" ]]; then
    "$python_bin" -m vgae_cf prepare --data "$DATA_DIR" 2>&1 | tee -a "$DATA_DIR/vgae_graph_preparation.log"
fi

#!/usr/bin/env bash
#SBATCH --job-name=uv-vgae
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
# Supply partition, GPU type, memory, wall time and --array through your site's sbatch options.
set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?Submit with --array=0-8 or --array=9-14}"
: "${REPO_DIR:?Set REPO_DIR to the checkout}"
cd "$REPO_DIR"
bash scripts/run_4gpu.sh "$SLURM_ARRAY_TASK_ID" "$@"

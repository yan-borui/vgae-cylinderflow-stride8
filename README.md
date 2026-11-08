# GLaDiT VGAE on MegaFlow2D Ellipse

This branch implements the original high-resolution mesh UVP task: file frames 1,16,…,961, one observation and 64 predictions, Train782 / Validation100 / Test100. All training uses four GPUs and checkpoint selection uses full Validation100.

Start with [ELLIPSE.md](ELLIPSE.md) for data preparation, training, recovery, budgets and verification status. Use [TEST.md](TEST.md) for final evaluation. Architecture and optimizer settings are inherited from the Airfoil four-GPU source; [AIRFOIL.md](AIRFOIL.md) is historical reference, and its launch commands do not define this branch’s task.

```bash
export RAW_DATA_DIR=/data/datasets/megaflow2d
export DATA_DIR=/data/datasets/megaflow2d_ellipse_stride15
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RESULT_ROOT=/data/runs/ellipse_vgae_seed0
bash scripts/ellipse_4gpu.sh train
```

## VGAE handoff

The fixed slot is `campaign/runs/w512_d4-4-2_c4_seed0`. Its Validation-selected export is `dit_autoencoder.pt`; set the DiT AUTOENCODER to this file. The export carries the selected UVP objective, representation ID, normalization and architecture. The four-GPU environment profile is retained across resume. Final saved reconstruction bundles contain the observed first frame plus 64 reconstructed frames; scoring excludes the observation.

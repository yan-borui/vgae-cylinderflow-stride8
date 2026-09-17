"""Personal single-GPU UVP VGAE on the released CylinderFlow stride-8 split."""

__version__ = "0.2.0"
PROTOCOL = "airfoil.uvp_vgae.four_gpu_locked.v1"
CODEC_FORMAT = "vgae_cf.airfoil_uvp_codec.v1"
DIT_AUTOENCODER_FORMAT = "vgae_cf.airfoil_uvp_dit_autoencoder.v1"
CHECKPOINT_FORMAT = "vgae_cf.airfoil_uvp_training_checkpoint.v1"

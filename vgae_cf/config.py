"""The fixed campaign design; this module has no ML runtime dependency."""

from __future__ import annotations


from . import PROTOCOL


RETRAIN_CONFIG_ID = "w512_d4-4-2_c4"
RETRAIN_SLOT = 0


RECIPE = {
    "world_size": 4,
    "microbatch": 4,
    "accumulation": 1,
    "precision": "fp32",
    "tf32": False,
    "activation_checkpointing": True,
    "batch_normalization": "synchronized_global_batch16",
    "optimizer": "Adam",
    "learning_rate": 1e-4,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "clip_norm": 1.0,
    "kl_weight": 1e-6,
    "validation_every": 10,
    "maximum_epochs": 5000,
    "scheduler": "ReduceLROnPlateau_train_epoch_mean_total",
    "lr_patience": 50,
    "lr_factor": 0.1,
    "minimum_lr": 0.0,
    "stopping_learning_rate": 1e-8,
    "reconstruction": "normalized_uvp_global_node_channel_mse",
    "selection": "validation24_uvp_total_loss",
    "coarse_level": 3,
    "train_frames": [0, 74],
    "evaluation_frames": [1, 64],
    "test_accessed": False,
}


def configurations() -> list[dict]:
    """Exactly the user's 5090 model; this branch has no tuning campaign."""
    return [{"id": RETRAIN_CONFIG_ID, "width": 512, "depths": [4, 4, 2], "latent": 4}]


def architecture(config: dict) -> dict:
    return {
        "in_node_features": 3,
        "cond_node_features": 4,
        "cond_edge_features": 2,
        "latent_node_features": config["latent"],
        "depths": list(config["depths"]),
        "fnns_depth": 2,
        "fnns_width": config["width"],
        "aggr": "sum",
        "dropout": 0.1,
        "norm_latents": False,
    }


def run_spec(config: dict, seed: int, stage: int) -> dict:
    return {
        "id": f"{config['id']}_seed{seed}",
        "config_id": config["id"],
        "config": config,
        "seed": seed,
        "stage": stage,
        "protocol": PROTOCOL,
        "architecture": architecture(config),
        "recipe": dict(RECIPE),
    }

"""The fixed campaign design; this module has no ML runtime dependency."""

from __future__ import annotations

from importlib.resources import files
import json

from . import PROTOCOL


RECIPE = {
    "world_size": 4,
    "microbatch": 4,
    "accumulation": 1,
    "precision": "fp32",
    "tf32": False,
    "optimizer": "Adam",
    "learning_rate": 1e-4,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "clip_norm": 1.0,
    "kl_weight": 1e-6,
    "validation_every": 10,
    "minimum_epochs": 500,
    "maximum_epochs": 5000,
    "lr_patience": 5,
    "lr_factor": 0.1,
    "minimum_lr": 1e-8,
    "early_stop_patience": 20,
    "selection": "validation24_uv_mse",
    "coarse_level": 3,
    "train_frames": [0, 74],
    "evaluation_frames": [1, 64],
    "test_accessed": False,
}


def configurations() -> list[dict]:
    """Return baseline first, then the eight deduplicated axis configurations."""
    design = json.loads(
        files("vgae_cf").joinpath("configs/scale_axes.json").read_text()
    )
    result = [design["baseline"]]
    seen = set()
    for axis, values in design["axes"].items():
        for value in values:
            item = {**design["center"], axis: value}
            key = (item["width"], tuple(item["depths"]), item["latent"])
            if key in seen:
                continue
            seen.add(key)
            depth_name = "-".join(str(n) for n in item["depths"])
            result.append(
                {"id": f"w{item['width']}_d{depth_name}_c{item['latent']}", **item}
            )
    if len(result) != 9 or len({row["id"] for row in result}) != 9:
        raise ValueError("the approved design must contain exactly nine configurations")
    return result


def architecture(config: dict) -> dict:
    return {
        "in_node_features": 2,
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

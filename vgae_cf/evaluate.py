"""Posterior-mean UVP reconstruction with equal frame and trajectory weighting."""

from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np
import torch

from .data import Dataset
from .config import RECIPE
from .distributed import Context, isolated_rng
from .io import write_json
from .metrics import aggregate, physical_metrics
from .model import UVPVGAE
from .objective import loss_components


def evaluate(
    model: UVPVGAE,
    data: Dataset,
    indices: tuple[int, ...],
    context: Context,
    directory: Path,
    *,
    save_fields: bool = False,
) -> dict:
    """Equal trajectory weighting, all stored frames 1..64, eval-mode BN."""
    context.primary_call(lambda: directory.mkdir(parents=True, exist_ok=True))
    was_training = model.training
    model.eval()
    mean, std = data.mean.to(context.device), data.std.to(context.device)

    def local_evaluation():
        rows = []
        with isolated_rng(context.device), torch.inference_mode():
            for index in indices[context.rank :: context.world]:
                geometry = data.geometry(index)
                reference = data.read_uvp(index, 1, 65)
                graph_template = data.static_graph(index).to(context.device)
                predictions, raw_predictions = [], []
                losses = []
                encode_seconds, decode_seconds = 0.0, 0.0
                for field in reference:
                    graph = graph_template.clone()
                    normalized = (
                        torch.from_numpy(field).to(context.device) - mean
                    ) / std
                    torch.cuda.synchronize(context.device)
                    started = time.perf_counter()
                    posterior = model.encode_latent(graph, normalized)
                    torch.cuda.synchronize(context.device)
                    encoded = time.perf_counter()
                    prediction, raw = model.decode_latent(
                        graph, posterior.mean, posterior.context
                    )
                    torch.cuda.synchronize(context.device)
                    decoded = time.perf_counter()
                    values = loss_components(
                        prediction,
                        normalized,
                        posterior.mean,
                        posterior.logvar,
                        RECIPE["kl_weight"],
                    )
                    losses.append([float(value) for value in values])
                    encode_seconds += encoded - started
                    decode_seconds += decoded - encoded
                    predictions.append((prediction * std + mean).cpu().numpy())
                    raw_predictions.append((raw * std + mean).cpu().numpy())
                prediction = np.stack(predictions)
                raw = np.stack(raw_predictions)
                total, reconstruction, kl = np.mean(losses, axis=0)
                metrics = physical_metrics(prediction, raw, reference, geometry)
                metrics.update(
                    uvp_total_loss=float(total),
                    normalized_uvp_mse=float(reconstruction),
                    kl_unweighted=float(kl),
                )
                row = {
                    "trajectory": index,
                    "frames": 64,
                    "nodes": len(geometry["points"]),
                    "latent_nodes": int(graph_template.pos_3.shape[0]),
                    "latent_channels": model.latent_node_features,
                    "latent_elements_per_frame": int(graph_template.pos_3.shape[0])
                    * model.latent_node_features,
                    "encode_seconds": encode_seconds,
                    "decode_seconds": decode_seconds,
                    "metrics": metrics,
                }
                write_json(directory / f"trajectory_{index:04d}.json", row)
                if save_fields:
                    destination = directory / f"trajectory_{index:04d}.npz"
                    temporary = destination.with_suffix(".tmp")
                    with temporary.open("wb") as stream:
                        np.savez_compressed(
                            stream,
                            prediction=prediction,
                            target=reference,
                            points=geometry["points"],
                            cells=geometry["cells"],
                            stored_frames=np.arange(1, 65, dtype=np.int64),
                            fields=np.asarray(["u", "v", "p"]),
                        )
                    os.replace(temporary, destination)
                rows.append(row)
                print(
                    f"rank={context.rank} eval trajectory={index} UVP-total={metrics['uvp_total_loss']:.8g} UV-MSE={metrics['uv_mse']:.8g}",
                    flush=True,
                )
        return rows

    try:
        rows = [row for shard in context.all_call(local_evaluation) for row in shard]
        rows.sort(key=lambda row: row["trajectory"])
        result = {
            "mode": "eval_posterior_mean",
            "stored_frames": [1, 64],
            "trajectory_indices": list(indices),
            "trajectory_count": len(indices),
            "fields": ["u", "v", "p"],
            "selection_metric": "uvp_total_loss",
            "kl_weight": RECIPE["kl_weight"],
            "weighting": "equal trajectories and frames; normalized UVP node/channel-mean MSE plus latent-element-mean KL per frame",
            "metrics": aggregate(rows, indices),
            "trajectories": rows,
            "encode_ms_per_frame": sum(row["encode_seconds"] for row in rows)
            * 1000
            / (64 * len(rows)),
            "decode_ms_per_frame": sum(row["decode_seconds"] for row in rows)
            * 1000
            / (64 * len(rows)),
            "timing": "synchronized CUDA wall time; single graph; encoder includes condition encoder; no I/O",
            "test_accessed": False,
        }
        context.primary_call(lambda: write_json(directory / "summary.json", result))
        return result
    finally:
        model.train(was_training)


def reconstruction_figure(file_name: Path, destination: Path) -> None:
    """Physical UVP with a shared target/prediction scale over three frames."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri

    with np.load(file_name) as payload:
        target, prediction = payload["target"], payload["prediction"]
        triangulation = tri.Triangulation(*payload["points"].T, payload["cells"])
    selected = [0, 31, 63]
    figure, axes = plt.subplots(6, 3, figsize=(13, 12), layout="constrained")
    for channel, name in enumerate(("u", "v", "p")):
        low = min(
            target[selected, :, channel].min(), prediction[selected, :, channel].min()
        )
        high = max(
            target[selected, :, channel].max(), prediction[selected, :, channel].max()
        )
        for column, frame in enumerate(selected):
            for offset, (label, values) in enumerate(
                (("reference", target), ("reconstruction", prediction))
            ):
                axis = axes[2 * channel + offset, column]
                artist = axis.tripcolor(
                    triangulation,
                    values[frame, :, channel],
                    shading="gouraud",
                    vmin=low,
                    vmax=high,
                )
                axis.set(title=f"{name} {label} / stored {frame + 1}", aspect="equal")
        figure.colorbar(
            artist,
            ax=axes[2 * channel : 2 * channel + 2].ravel().tolist(),
            label=f"physical {name}",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def final_figures(summary: dict, directory: Path) -> None:
    ordered = sorted(
        summary["trajectories"], key=lambda row: row["metrics"]["uvp_total_loss"]
    )
    for label, row in (
        ("best", ordered[0]),
        ("median", ordered[len(ordered) // 2]),
        ("worst", ordered[-1]),
    ):
        index = row["trajectory"]
        reconstruction_figure(
            directory / f"trajectory_{index:04d}.npz",
            directory / f"{label}_{index:04d}.png",
        )

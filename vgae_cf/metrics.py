"""Physical UV metrics; derivatives use linear interpolation on each triangle."""

from __future__ import annotations

import math

import numpy as np


def triangle_fields(
    velocity: np.ndarray, points: np.ndarray, cells: np.ndarray
) -> tuple:
    p = np.asarray(points, dtype=np.float64)[cells]
    x0, y0 = p[:, 0, 0], p[:, 0, 1]
    x1, y1 = p[:, 1, 0], p[:, 1, 1]
    x2, y2 = p[:, 2, 0], p[:, 2, 1]
    determinant = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if np.any(np.abs(determinant) <= np.finfo(np.float64).eps):
        raise ValueError("mesh contains a degenerate triangle")
    values = np.asarray(velocity, dtype=np.float64)[:, cells]
    dx = (
        values[:, :, 0] * (y1 - y2)[None, :, None]
        + values[:, :, 1] * (y2 - y0)[None, :, None]
        + values[:, :, 2] * (y0 - y1)[None, :, None]
    ) / determinant[None, :, None]
    dy = (
        values[:, :, 0] * (x2 - x1)[None, :, None]
        + values[:, :, 1] * (x0 - x2)[None, :, None]
        + values[:, :, 2] * (x1 - x0)[None, :, None]
    ) / determinant[None, :, None]
    return dx[..., 1] - dy[..., 0], dx[..., 0] + dy[..., 1], 0.5 * np.abs(determinant)


def physical_metrics(
    prediction: np.ndarray, raw: np.ndarray, target: np.ndarray, geometry: dict
) -> dict:
    if prediction.shape != target.shape or target.ndim != 3 or target.shape[-1] != 2:
        raise ValueError("metrics require aligned physical UV [frames, nodes, 2]")
    if not all(np.isfinite(item).all() for item in (prediction, raw, target)):
        raise ValueError("nonfinite prediction or reference; evaluation is invalid")
    prediction, raw, target = (
        np.asarray(item, dtype=np.float64) for item in (prediction, raw, target)
    )
    points, cells, labels = (geometry[key] for key in ("points", "cells", "node_type"))
    pred_vorticity, pred_divergence, area = triangle_fields(prediction, points, cells)
    true_vorticity, true_divergence, _ = triangle_fields(target, points, cells)
    weights = np.zeros(len(points), dtype=np.float64)
    for vertex in range(3):
        np.add.at(weights, cells[:, vertex], area / 3)
    if np.any(weights <= 0):
        raise ValueError("every node must have positive incident triangle area")
    error = prediction - target
    error_energy = float(np.sum(error**2 * weights[None, :, None]))
    target_energy = float(np.sum(target**2 * weights[None, :, None]))
    triangle_denominator = area.sum() * len(target)

    def area_rms(values):
        return float(np.sqrt(np.sum(values**2 * area) / triangle_denominator))

    result = {
        "uv_mse": float(np.mean(error**2)),
        "u_mse": float(np.mean(error[..., 0] ** 2)),
        "v_mse": float(np.mean(error[..., 1] ** 2)),
        "area_uv_relative_rmse": math.sqrt(error_energy / target_energy)
        if target_energy > 0
        else None,
        "vorticity_rmse": area_rms(pred_vorticity - true_vorticity),
        "divergence_rmse": area_rms(pred_divergence - true_divergence),
        "predicted_divergence_rms": area_rms(pred_divergence),
        "reference_divergence_rms": area_rms(true_divergence),
    }
    for name, mask in {
        "inlet": labels == 4,
        "wall": labels == 6,
        "outlet": labels == 5,
        "boundary": np.isin(labels, [4, 5, 6]),
    }.items():
        result[f"{name}_uv_rmse"] = (
            float(np.sqrt(np.mean(error[:, mask] ** 2))) if mask.any() else None
        )
        result[f"{name}_raw_uv_rmse"] = (
            float(np.sqrt(np.mean((raw[:, mask] - target[:, mask]) ** 2)))
            if mask.any()
            else None
        )
    return result


def aggregate(rows: list[dict], expected_indices: tuple[int, ...]) -> dict:
    if sorted(row["trajectory"] for row in rows) != sorted(expected_indices):
        raise ValueError(
            "evaluation must cover every requested trajectory exactly once"
        )
    result = {}
    for name in rows[0]["metrics"]:
        values = [row["metrics"][name] for row in rows]
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError(f"invalid evaluation metric: {name}")
        valid = [value for value in values if value is not None]
        result[name] = float(np.mean(valid)) if valid else None
        result[name + "_count"] = len(valid)
    if result["uv_mse_count"] != len(expected_indices):
        raise ValueError("selection MSE cannot exclude an invalid trajectory")
    return result

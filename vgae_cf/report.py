"""Screening curves and paired three-seed confirmation summaries."""

from __future__ import annotations

import csv
from pathlib import Path
import statistics

from . import PROTOCOL
from .campaign import completed_result
from .io import read_json, write_json


def write_csv(destination: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
    }


def render_curves(results: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    screen = [result for result in results if result["spec"]["stage"] == 1]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for axis, dimension in zip(axes, ("width", "depths", "latent")):
        rows = []
        for result in screen:
            config = result["spec"]["config"]
            center = {"width": 512, "depths": [4, 4, 2], "latent": 4}
            if all(
                config[key] == value
                for key, value in center.items()
                if key != dimension
            ):
                x_value = (
                    sum(config[dimension])
                    if dimension == "depths"
                    else config[dimension]
                )
                rows.append(
                    (x_value, result["validation24"]["metrics"]["uvp_total_loss"])
                )
        if rows:
            rows.sort()
            axis.plot(*zip(*rows), "o-")
            axis.set_xticks([item[0] for item in rows])
        axis.set(
            xlabel="encoder total depth (5/10/20)"
            if dimension == "depths"
            else dimension,
            ylabel="Validation-24 normalized UVP total loss",
            title=f"{dimension} / seed 0 screening",
        )
        axis.grid(alpha=0.2)
    figure.savefig(output / "scaling_axes.png", dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for result in screen:
        score = result["validation24"]["metrics"]["uvp_total_loss"]
        axes[0].scatter(result["parameter_count"] / 1e6, score)
        axes[0].annotate(
            result["spec"]["config_id"],
            (result["parameter_count"] / 1e6, score),
            fontsize=6,
        )
        trajectory_rows = result["validation24"]["trajectories"]
        capacity = statistics.mean(
            row["latent_elements_per_frame"] for row in trajectory_rows
        )
        axes[1].scatter(capacity, score)
        exposure = 0
        exposure_curve, loss_curve = [], []
        for epoch in result["history"]:
            exposure += epoch["sampled_frames"]
            if "validation24_uvp_total_loss" in epoch:
                exposure_curve.append(exposure)
                loss_curve.append(epoch["validation24_uvp_total_loss"])
        axes[2].plot(
            exposure_curve, loss_curve, label=result["spec"]["config_id"], linewidth=1
        )
    for axis, label in zip(
        axes,
        (
            "parameters / million",
            "mean latent elements / frame",
            "training frame exposures",
        ),
    ):
        axis.set(xlabel=label, ylabel="Validation-24 normalized UVP total loss")
        axis.grid(alpha=0.2)
    axes[2].legend(fontsize=5)
    figure.savefig(output / "capacity_and_exposure.png", dpi=170)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    for result in screen:
        gpu_seconds = 0.0
        xs, ys = [], []
        for epoch in result["history"]:
            gpu_seconds += epoch["training_gpu_seconds"]
            if "validation24_uvp_total_loss" in epoch:
                xs.append(gpu_seconds / 3600)
                ys.append(epoch["validation24_uvp_total_loss"])
        axis.plot(xs, ys, label=result["spec"]["config_id"])
    axis.set(
        xlabel="training GPU hours (1 x elapsed training)",
        ylabel="Validation-24 normalized UVP total loss",
    )
    axis.legend(fontsize=6)
    axis.grid(alpha=0.2)
    figure.savefig(output / "gpu_cost.png", dpi=170)
    plt.close(figure)


def compare_fields(campaign: Path, selected: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    import numpy as np

    baseline = selected["baseline"][0]
    ordered = sorted(
        baseline["validation100"]["trajectories"],
        key=lambda row: row["metrics"]["uvp_total_loss"],
    )
    cases = (
        ("baseline_best", ordered[0]),
        ("baseline_median", ordered[len(ordered) // 2]),
        ("baseline_worst", ordered[-1]),
    )
    for case_name, row in cases:
        index = row["trajectory"]
        arrays = []
        for config_id, seed_results in selected.items():
            for seed, result in enumerate(seed_results):
                source = (
                    campaign
                    / "runs"
                    / result["spec"]["id"]
                    / result["final_evaluation_directory"]
                    / f"trajectory_{index:04d}.npz"
                )
                with np.load(source) as payload:
                    arrays.append((f"{config_id}/s{seed}", payload["prediction"][31]))
                    reference, points, cells = (
                        payload["target"][31],
                        payload["points"],
                        payload["cells"],
                    )
        arrays.insert(0, ("reference", reference))
        figure, axes = plt.subplots(
            len(arrays), 3, figsize=(16, 2 * len(arrays)), layout="constrained"
        )
        triangulation = tri.Triangulation(*points.T, cells)
        for channel, field_name in enumerate(("u", "v", "p")):
            low = min(values[:, channel].min() for _, values in arrays)
            high = max(values[:, channel].max() for _, values in arrays)
            for position, (label, values) in enumerate(arrays):
                artist = axes[position, channel].tripcolor(
                    triangulation,
                    values[:, channel],
                    shading="gouraud",
                    vmin=low,
                    vmax=high,
                )
                axes[position, channel].set(
                    title=f"{label} / {field_name}", aspect="equal"
                )
            figure.colorbar(
                artist, ax=axes[:, channel].tolist(), label=f"physical {field_name}"
            )
        figure.suptitle(
            f"trajectory {index}, stored frame 32; shared scale across all configurations and seeds"
        )
        figure.savefig(output / f"paired_{case_name}_{index:04d}.png", dpi=130)
        plt.close(figure)


def report(campaign: Path, output: Path) -> dict:
    plan = read_json(campaign / "campaign.json")
    if plan.get("protocol") != PROTOCOL:
        raise ValueError("report requires a current UVP campaign")
    output.mkdir(parents=True, exist_ok=True)
    results, pending = [], []
    for spec in plan["runs"]:
        if (campaign / "runs" / spec["id"] / "result.json").is_file():
            results.append(completed_result(campaign, spec))
        else:
            pending.append(spec["id"])
    table = []
    for result in results:
        config = result["spec"]["config"]
        final = result["validation100"]
        table.append(
            {
                "run": result["spec"]["id"],
                "stage": result["spec"]["stage"],
                "seed": result["spec"]["seed"],
                "width": config["width"],
                "encoder_depths": "/".join(map(str, config["depths"])),
                "latent_channels": config["latent"],
                "parameters": result["parameter_count"],
                "best_epoch": result["best_epoch"],
                "training_epochs": result["training_epochs"],
                "training_exposures": result["training_exposures"],
                "validation24_uvp_total_loss": result["validation24"]["metrics"][
                    "uvp_total_loss"
                ],
                "validation100_uvp_total_loss": final["metrics"]["uvp_total_loss"],
                "validation24_uv_mse": result["validation24"]["metrics"]["uv_mse"],
                "validation100_uv_mse": final["metrics"]["uv_mse"],
                "pressure_raw_rmse": final["metrics"]["pressure_raw_rmse"],
                "pressure_gauge_free_rmse": final["metrics"][
                    "pressure_gauge_free_rmse"
                ],
                "area_uv_relative_rmse": final["metrics"]["area_uv_relative_rmse"],
                "vorticity_rmse": final["metrics"]["vorticity_rmse"],
                "divergence_rmse": final["metrics"]["divergence_rmse"],
                "boundary_uv_rmse": final["metrics"]["boundary_uv_rmse"],
                "boundary_raw_uv_rmse": final["metrics"]["boundary_raw_uv_rmse"],
                "latent_elements_mean": statistics.mean(
                    row["latent_elements_per_frame"] for row in final["trajectories"]
                ),
                "training_gpu_hours": result["training_gpu_seconds"] / 3600,
                "committed_gpu_hours": result["committed_gpu_seconds"] / 3600,
                "peak_allocated_gib": max(result["peak_allocated_bytes_by_rank"])
                / 2**30,
                "encode_ms_per_frame": final["encode_ms_per_frame"],
                "decode_ms_per_frame": final["decode_ms_per_frame"],
            }
        )
    write_csv(output / "runs.csv", table)
    selected = {}
    for config_id in plan.get("confirmation_config_ids", []):
        seeds = sorted(
            [item for item in results if item["spec"]["config_id"] == config_id],
            key=lambda item: item["spec"]["seed"],
        )
        if [item["spec"]["seed"] for item in seeds] == [0, 1, 2]:
            selected[config_id] = seeds
    summary = {
        "screening": "seed 0 only; candidate selection, not a stability claim",
        "pending": pending,
        "completed_formal_runs": len(results),
        "maximum_formal_runs": 15,
        "three_seed": {},
        "paired": {},
        "test_accessed": False,
    }
    for config_id, seeds in selected.items():
        summary["three_seed"][config_id] = {
            split: {
                metric: mean_std([item[split]["metrics"][metric] for item in seeds])
                for metric in seeds[0][split]["metrics"]
                if not metric.endswith("_count")
                and all(item[split]["metrics"][metric] is not None for item in seeds)
            }
            for split in ("validation24", "validation100")
        }
    paired_rows = []
    if "baseline" in selected:
        for config_id, seeds in selected.items():
            if config_id == "baseline":
                continue
            summary["paired"][config_id] = {}
            for split in ("validation24", "validation100"):
                changes = []
                for seed, (candidate, baseline) in enumerate(
                    zip(seeds, selected["baseline"])
                ):
                    score, base = (
                        item[split]["metrics"]["uvp_total_loss"]
                        for item in (candidate, baseline)
                    )
                    changes.append(score - base)
                    paired_rows.append(
                        {
                            "config": config_id,
                            "seed": seed,
                            "split": split,
                            "candidate_uvp_total_loss": score,
                            "baseline_uvp_total_loss": base,
                            "delta_uvp_total_loss": score - base,
                            "relative_change_percent": 100 * (score / base - 1)
                            if base
                            else None,
                        }
                    )
                summary["paired"][config_id][split] = mean_std(changes)
    write_csv(output / "paired_seeds.csv", paired_rows)
    complete = len(selected) == 3 and len(results) == 15 and not pending
    if complete:
        summary["selected_by_mean_validation24"] = min(
            selected,
            key=lambda key: (
                summary["three_seed"][key]["validation24"]["uvp_total_loss"]["mean"],
                key,
            ),
        )
    write_json(output / "summary.json", summary)
    if results:
        render_curves(results, output)
    if complete:
        compare_fields(campaign, selected, output)
    lines = [
        "# VGAE 扩容结果",
        "",
        f"已完成 {len(results)}/15 次正式训练。Test 保持封存。",
        "",
        "首轮是 seed0 筛选；最终配置按三 seed 的平均 Validation-24 UVP 总 loss 选择。",
        "Validation-100 是所选权重的补充评价，未参与学习率、早停或首轮晋级。",
        "",
        "三条线描述交点附近的单轴变化；未估计宽度、深度、latent 的交互作用。",
        "网络扩容收益由宽度线、深度线判断；通道容量收益由 latent 线判断。",
        "固定的是网络和数据配方；自适应收敛轮数不同，需同时查看曝光量和 GPU 时间。",
        "",
        "|配置|Validation-24 总 loss（均值 ± 样本标准差）|Validation-100 总 loss（均值 ± 样本标准差）|",
        "|---|---|---|",
    ]
    for config_id, metrics in summary["three_seed"].items():
        values = [
            metrics[split]["uvp_total_loss"]
            for split in ("validation24", "validation100")
        ]
        lines.append(
            f"|{config_id}|{values[0]['mean']:.8g} ± {values[0]['std']:.3g}|{values[1]['mean']:.8g} ± {values[1]['std']:.3g}|"
        )
    if complete:
        lines += [
            "",
            f"按预定指标选择：**{summary['selected_by_mean_validation24']}**。",
            "",
            "配对变化见 paired_seeds.csv；负值表示低于同 seed 基线。仅 3 个 seed，不作显著性保证。",
        ]
    else:
        lines += ["", "重复训练尚未全部完成；当前不形成最终稳定性或胜出结论。"]
    lines += [
        "",
        "成本、参数量、latent 元素数、辅助指标和编解码耗时见 runs.csv；逐轨迹原始记录保存在各 run 的 evaluation 目录。",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary

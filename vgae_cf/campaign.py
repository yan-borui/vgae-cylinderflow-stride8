"""Immutable 9 + 6 campaign slots; continuations retain the same run identity."""

from __future__ import annotations

import math
from pathlib import Path

from . import PROTOCOL
from .config import configurations, run_spec
from .io import file_lock, read_json, source_identity, write_json


def create(directory: Path) -> dict:
    with file_lock(directory):
        destination = directory / "campaign.json"
        if destination.exists():
            return read_json(destination)
        rows = [run_spec(config, 0, 1) for config in configurations()]
        for slot, spec in enumerate(rows):
            spec["slot"] = slot
        result = {
            "protocol": PROTOCOL,
            "maximum_formal_runs": 15,
            "stage": 1,
            "runs": rows,
            "source": source_identity(),
            "test_accessed": False,
        }
        write_json(destination, result)
        return result


def slot(directory: Path, number: int) -> tuple[dict, dict]:
    plan = read_json(directory / "campaign.json")
    if plan["protocol"] != PROTOCOL or not 0 <= number < len(plan["runs"]):
        raise ValueError("slot is outside the approved campaign")
    if len(plan["runs"]) not in (9, 15) or plan["maximum_formal_runs"] != 15:
        raise ValueError("campaign must contain nine or fifteen formal runs")
    if len({item["id"] for item in plan["runs"]}) != len(plan["runs"]):
        raise ValueError("each formal slot must have a unique run identity")
    expected_screen = [
        {**run_spec(config, 0, 1), "slot": i}
        for i, config in enumerate(configurations())
    ]
    if plan["runs"][:9] != expected_screen:
        raise ValueError("the nine screening configurations are fixed")
    spec = plan["runs"][number]
    configs = {config["id"]: config for config in configurations()}
    if spec["config_id"] not in configs:
        raise ValueError("unknown configuration")
    expected = run_spec(configs[spec["config_id"]], spec["seed"], spec["stage"])
    if spec != {**expected, "slot": number}:
        raise ValueError("campaign settings differ from the approved recipe")
    if (number < 9 and (spec["seed"], spec["stage"]) != (0, 1)) or (
        number >= 9 and (spec["seed"] not in (1, 2) or spec["stage"] != 2)
    ):
        raise ValueError("invalid stage/seed assignment")
    return spec, plan


def completed_result(directory: Path, spec: dict) -> dict:
    result = read_json(directory / "runs" / spec["id"] / "result.json")
    if (
        result["status"] != "completed"
        or result["mode"] != "formal"
        or result["spec"] != spec
    ):
        raise ValueError(f"{spec['id']} has no completed matching formal result")
    if (
        result["validation24"]["trajectory_count"] != 24
        or result["validation100"]["trajectory_count"] != 100
    ):
        raise ValueError("formal result lacks the required Validation coverage")
    score = result["validation24"]["metrics"]["uv_mse"]
    if not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("invalid Validation-24 selection MSE")
    best = read_json(directory / "runs" / spec["id"] / "best.json")
    if (
        best["checkpoint"] != result["best_checkpoint"]
        or best["epoch"] != result["best_epoch"]
        or best["validation24_uv_mse"] != score
        or result.get("test_accessed") is not False
    ):
        raise ValueError("reported result and selected checkpoint index disagree")
    for split, count in (("validation24", 24), ("validation100", 100)):
        evaluation = result[split]
        if (
            evaluation["stored_frames"] != [1, 64]
            or evaluation["metrics"]["uv_mse_count"] != count
            or len(set(evaluation["trajectory_indices"])) != count
            or evaluation["mode"] != "eval_posterior_mean"
        ):
            raise ValueError("reported evaluation differs from the selection protocol")
    return result


def confirm(directory: Path) -> dict:
    with file_lock(directory):
        plan = read_json(directory / "campaign.json")
        if plan["stage"] == 2:
            return plan
        if len(plan["runs"]) != 9:
            raise ValueError("screening must contain exactly nine runs")
        results = []
        for number in range(9):
            spec, _ = slot(directory, number)
            results.append(completed_result(directory, spec))
        expanded = [
            result for result in results if result["spec"]["config_id"] != "baseline"
        ]
        expanded.sort(
            key=lambda result: (
                result["validation24"]["metrics"]["uv_mse"],
                result["spec"]["config_id"],
            )
        )
        chosen = ["baseline"] + [result["spec"]["config_id"] for result in expanded[:2]]
        by_id = {config["id"]: config for config in configurations()}
        for config_id in chosen:
            for seed in (1, 2):
                spec = run_spec(by_id[config_id], seed, 2)
                spec["slot"] = len(plan["runs"])
                plan["runs"].append(spec)
        plan["stage"] = 2
        plan["confirmation_config_ids"] = chosen
        plan["screening_selection"] = [
            {
                "id": result["spec"]["id"],
                "validation24_uv_mse": result["validation24"]["metrics"]["uv_mse"],
                "best_checkpoint": result["best_checkpoint"],
                "best_epoch": result["best_epoch"],
            }
            for result in results
        ]
        if len(plan["runs"]) != 15:
            raise ValueError("confirmation must bring the total to exactly fifteen")
        write_json(directory / "campaign.json", plan)
        return plan


def status(directory: Path) -> list[dict]:
    plan = read_json(directory / "campaign.json")
    rows = []
    for spec in plan["runs"]:
        root = directory / "runs" / spec["id"]
        marker = root / "status.json"
        row = read_json(marker) if marker.exists() else {"status": "pending"}
        rows.append({"slot": spec["slot"], "run": spec["id"], **row})
    return rows

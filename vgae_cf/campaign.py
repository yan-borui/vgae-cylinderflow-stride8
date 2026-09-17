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
        expected = {**run_spec(configurations()[0], 0, 1), "slot": 0}
        if destination.exists():
            result = read_json(destination)
            slot(directory, 0)
            return result
        result = {
            "protocol": PROTOCOL,
            "maximum_formal_runs": 1,
            "stage": 1,
            "runs": [expected],
            "source": source_identity(),
            "test_accessed": False,
        }
        write_json(destination, result)
        return result


def slot(directory: Path, number: int) -> tuple[dict, dict]:
    plan = read_json(directory / "campaign.json")
    expected = {**run_spec(configurations()[0], 0, 1), "slot": 0}
    if (
        number != 0
        or plan.get("protocol") != PROTOCOL
        or plan.get("maximum_formal_runs") != 1
        or plan.get("runs") != [expected]
    ):
        raise ValueError("Airfoil locks one w512_d4-4-2_c4 seed0 run in slot0")
    return expected, plan


def retrain_slot(directory: Path) -> int:
    slot(directory, 0)
    return 0


def completed_result(directory: Path, spec: dict) -> dict:
    if spec.get("protocol") != PROTOCOL:
        raise ValueError("result belongs to a different VGAE protocol")
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
    score = result["validation24"]["metrics"]["uvp_total_loss"]
    if not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("invalid Validation-24 UVP total loss")
    best = read_json(directory / "runs" / spec["id"] / "best.json")
    if (
        best["checkpoint"] != result["best_checkpoint"]
        or best["epoch"] != result["best_epoch"]
        or best["validation24_uvp_total_loss"] != score
        or result["validation24_uvp_total_loss"] != score
        or result.get("test_accessed") is not False
    ):
        raise ValueError("reported result and selected checkpoint index disagree")
    for split, count in (("validation24", 24), ("validation100", 100)):
        evaluation = result[split]
        if (
            evaluation["stored_frames"] != [1, 64]
            or evaluation["metrics"]["uvp_total_loss_count"] != count
            or evaluation.get("selection_metric") != "uvp_total_loss"
            or evaluation.get("fields") != ["u", "v", "p"]
            or len(set(evaluation["trajectory_indices"])) != count
            or evaluation["mode"] != "eval_posterior_mean"
        ):
            raise ValueError("reported evaluation differs from the selection protocol")
    return result


def status(directory: Path) -> list[dict]:
    plan = read_json(directory / "campaign.json")
    if plan.get("protocol") != PROTOCOL:
        raise ValueError("campaign belongs to a different VGAE protocol")
    rows = []
    for spec in plan["runs"]:
        root = directory / "runs" / spec["id"]
        marker = root / "status.json"
        row = read_json(marker) if marker.exists() else {"status": "pending"}
        rows.append({"slot": spec["slot"], "run": spec["id"], **row})
    return rows

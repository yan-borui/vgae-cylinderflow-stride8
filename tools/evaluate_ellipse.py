"""Four-GPU Validation/Test reconstruction of the already-selected UVP VGAE."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from vgae_cf import CHECKPOINT_FORMAT, PROTOCOL
from vgae_cf.data import Dataset
from vgae_cf.distributed import Context
from vgae_cf.evaluate import evaluate, final_figures
from vgae_cf.io import read_json, source_identity, write_json
from vgae_cf.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path, required=True, help="original training data directory"
    )
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    context = Context()
    try:
        environment = context.verify_environment(args.environment)
        selected = context.primary_call(lambda: read_json(args.run / "best.json"))
        checkpoint = args.run / selected["checkpoint"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata = payload["metadata"]
        if (
            payload.get("format") != CHECKPOINT_FORMAT
            or metadata["spec"].get("protocol") != PROTOCOL
        ):
            raise ValueError("Test reconstruction requires the matching UVP checkpoint")
        selection = payload["state"]["selection"]
        if (
            selected["checkpoint"] != selection["best_checkpoint"]
            or selected["epoch"] != selection["best_epoch"]
            or selected["selection_validation100_uvp_total_loss"]
            != selection["best_total_loss"]
        ):
            raise ValueError(
                "best.json differs from the stored Validation UVP selection"
            )
        training = Dataset(args.data)
        if (
            metadata["mode"] != "formal"
            or metadata["environment"] != environment
            or metadata["data"] != training.identity()
        ):
            raise ValueError("Test requires the original formal data and environment")
        if metadata["source"]["files"] != source_identity()["files"]:
            raise ValueError("core VGAE source differs from the selected checkpoint")
        context.primary_call(lambda: args.output.mkdir(parents=True, exist_ok=False))
        data = training
        identity = {
            "evaluation_split": args.split,
            "test_accessed": args.split == "test",
            "task": "posterior-mean reconstruction of stored frames 1..64",
            "checkpoint": str(checkpoint.resolve()),
            "selection": selected,
            "training_data_identity": training.identity(),
            "evaluation_data_identity": data.identity(),
        }
        context.primary_call(
            lambda: write_json(args.output / "evaluation_identity.json", identity)
        )
        model = build_model(metadata["spec"]["architecture"], context.device)
        model.load_state_dict(payload["model"], strict=True)
        del payload
        summary = evaluate(
            model, data, data.splits[args.split], context, args.output, save_fields=True
        )
        summary.update(identity)
        context.primary_call(lambda: write_json(args.output / "summary.json", summary))
        context.primary_call(lambda: final_figures(summary, args.output))
    finally:
        context.close()


if __name__ == "__main__":
    main()

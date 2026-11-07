"""Four-GPU Test reconstruction of the already-selected UVP VGAE."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mgn_data.test_data import TestData
from vgae_cf import CHECKPOINT_FORMAT, PROTOCOL
from vgae_cf.data import Dataset, FORMAT
from vgae_cf.distributed import Context
from vgae_cf.evaluate import evaluate, final_figures
from vgae_cf.io import read_json, source_identity, write_json
from vgae_cf.model import build_model


class ReconstructionData(Dataset):
    """Use Test geometry with frozen training normalization and a separate cache."""

    def __init__(self, training: Dataset, directory: Path, cache: Path):
        self.test = TestData(
            directory, "airfoil" if "airfoil" in FORMAT else "cylinderflow"
        )
        self.test.bind_training_manifest(training.manifest)
        self.directory = self.test.directory
        self.dataset_file = self.test.dataset
        self.manifest = self.test.manifest
        self.splits = self.test.splits
        self.normalization = training.normalization
        self.mean, self.std = training.mean, training.std
        self.graph_directory = cache
        self._graphs = {}

    def _check_index(self, index: int) -> None:
        if index not in self.splits["test"]:
            raise ValueError("trajectory is outside the official Test100")

    def read_uvp(self, index: int, start: int = 0, stop: int = 65):
        if not 0 <= start < stop <= 65:
            raise ValueError("Test reconstruction reads only stored frames 0..64")
        return super().read_uvp(index, start, stop)

    def identity(self) -> dict:
        return self.test.identity()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path, required=True, help="original training data directory"
    )
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--test-data-dir", type=Path, required=True)
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
            or selected["validation24_uvp_total_loss"] != selection["best_total_loss"]
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
        data = ReconstructionData(
            training, args.test_data_dir, args.output / "test_graphs"
        )
        identity = {
            "evaluation_split": "test",
            "test_accessed": True,
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
            model, data, data.splits["test"], context, args.output, save_fields=True
        )
        summary.update(identity)
        context.primary_call(lambda: write_json(args.output / "summary.json", summary))
        context.primary_call(lambda: final_figures(summary, args.output))
    finally:
        context.close()


if __name__ == "__main__":
    main()

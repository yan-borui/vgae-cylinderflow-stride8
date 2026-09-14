"""Administrative commands import no model until a GPU operation is requested."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UV VGAE: three scaling axes, fifteen formal runs"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="download pinned Train/Validation and freeze graph caches"
    )
    prepare.add_argument("--data", type=Path, required=True)
    prepare.add_argument("--download-only", action="store_true")
    campaign = commands.add_parser(
        "campaign", help="create/status/confirm the fixed campaign"
    )
    campaign.add_argument("action", choices=("create", "status", "confirm"))
    campaign.add_argument("--root", type=Path, default=Path("runs/campaign"))
    environment = commands.add_parser(
        "environment", help="record the intended formal four-GPU environment"
    )
    environment.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train", help="official training and resume path")
    train.add_argument("--campaign", type=Path, required=True)
    train.add_argument("--slot", type=int, required=True)
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--environment", type=Path, required=True)
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--mode", choices=("formal", "acceptance"), default="formal")
    train.add_argument("--acceptance-root", type=Path, default=Path("runs/acceptance"))
    restart = train.add_mutually_exclusive_group()
    restart.add_argument("--resume", action="store_true")
    restart.add_argument("--retry-initial", action="store_true")
    train.add_argument(
        "--stop-after-epoch",
        type=int,
        help="operator-requested pause after saving a completed epoch",
    )
    evaluation = commands.add_parser(
        "evaluate", help="reevaluate the selected checkpoint on full Validation"
    )
    evaluation.add_argument("--run", type=Path, required=True)
    evaluation.add_argument("--data", type=Path, required=True)
    evaluation.add_argument("--environment", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    report_parser = commands.add_parser(
        "report", help="render screening and paired three-seed results"
    )
    report_parser.add_argument("--campaign", type=Path, required=True)
    report_parser.add_argument("--output", type=Path, default=Path("artifacts/report"))
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare as prepare_data

        prepare_data(args.data, download_only=args.download_only)
    elif args.command == "campaign":
        from . import campaign as campaign_module

        result = getattr(campaign_module, args.action)(args.root)
        if args.action != "status":
            result = {key: value for key, value in result.items() if key != "source"}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "train":
        if args.workers < 0 or (
            args.stop_after_epoch is not None and args.stop_after_epoch < 1
        ):
            parser.error("workers must be nonnegative and stop-after-epoch positive")
        from .train import run

        run(args)
    elif args.command == "environment":
        from .distributed import Context
        from .io import read_json, write_json

        context = Context()
        try:
            observed = context.environment()

            def save():
                if args.output.exists() and read_json(args.output) != observed:
                    raise ValueError(
                        "existing environment profile differs; choose a new output file"
                    )
                write_json(args.output, observed)

            context.primary_call(save)
        finally:
            context.close()
    elif args.command == "evaluate":
        evaluate_selected(args)
    elif args.command == "report":
        from .report import report

        print(
            json.dumps(report(args.campaign, args.output), ensure_ascii=False, indent=2)
        )


def evaluate_selected(args) -> None:
    import torch
    from .data import Dataset
    from .distributed import Context
    from .evaluate import evaluate, final_figures
    from .io import read_json, source_identity
    from .model import build_model

    context = Context()
    try:
        environment = context.verify_environment(args.environment)
        index = context.primary_call(lambda: read_json(args.run / "best.json"))
        payload = torch.load(
            args.run / index["checkpoint"], map_location="cpu", weights_only=True
        )
        data = Dataset(args.data)
        metadata = payload["metadata"]
        if metadata["mode"] != "formal":
            raise ValueError(
                "this command requires a formal run; acceptance artifacts remain separate"
            )
        if (
            metadata["environment"] != environment
            or metadata["data"] != data.identity()
        ):
            raise ValueError("reevaluation requires the original data and environment")
        if metadata["source"]["files"] != source_identity()["files"]:
            raise ValueError("reevaluation source differs from the selected checkpoint")
        context.primary_call(lambda: args.output.mkdir(parents=True, exist_ok=False))
        model = build_model(metadata["spec"]["architecture"], context.device)
        model.load_state_dict(payload["model"], strict=True)
        del payload
        summary = evaluate(
            model,
            data,
            data.splits["validation"],
            context,
            args.output,
            save_fields=True,
        )
        context.primary_call(lambda: final_figures(summary, args.output))
    finally:
        context.close()


if __name__ == "__main__":
    main()

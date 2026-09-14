"""Official four-GPU training entrypoint, also used by the one acceptance flow."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from dgn4cfd.loader import Collater
from .campaign import slot
from .data import Dataset, EpochFrames, monitor_indices
from .distributed import Context, capture_rng, restore_rng, seed_everything
from .evaluate import evaluate, final_figures
from .io import file_lock, read_json, source_identity, write_json
from .model import build_model


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_checkpoint(destination: Path, payload: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def next_attempt(directory: Path, prefix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    number = 1
    while (directory / f"{prefix}_{number:03d}").exists():
        number += 1
    result = directory / f"{prefix}_{number:03d}"
    result.mkdir()
    return result


def epoch_batches(
    seed: int, epoch: int, rank: int, *, acceptance: bool
) -> list[list[int]]:
    generator = np.random.default_rng(np.random.SeedSequence([seed, epoch, 1100]))
    order = generator.permutation(1000).tolist()
    if acceptance:
        order = order[
            :24
        ]  # One full global batch and the same 8-graph tail as formal training.
    return [order[start : start + 16][rank::4] for start in range(0, len(order), 16)]


def update_selection(
    state: dict, score: float, epoch: int, recipe: dict
) -> tuple[dict, bool]:
    if not math.isfinite(score):
        raise ValueError("nonfinite Validation MSE is not a valid selection event")
    state = dict(state)
    improved = state["best_mse"] is None or score < state["best_mse"]
    if improved:
        state.update(
            best_mse=score, best_epoch=epoch, bad_evaluations=0, lr_bad_evaluations=0
        )
    else:
        state["bad_evaluations"] += 1
        state["lr_bad_evaluations"] += 1
        if state["lr_bad_evaluations"] >= recipe["lr_patience"]:
            state["learning_rate"] = max(
                recipe["minimum_lr"], state["learning_rate"] * recipe["lr_factor"]
            )
            state["lr_bad_evaluations"] = 0
    state["valid_evaluations"] += 1
    return state, improved


def train_epoch(
    model, optimizer, data, spec, epoch, context, workers, acceptance
) -> dict:
    model.train()
    generator = torch.Generator().manual_seed(spec["seed"] * 10000 + epoch)
    options = {"multiprocessing_context": "spawn"} if workers else {}
    loader = DataLoader(
        EpochFrames(data, spec["seed"], epoch),
        batch_sampler=epoch_batches(
            spec["seed"], epoch, context.rank, acceptance=acceptance
        ),
        collate_fn=Collater(),
        num_workers=workers,
        generator=generator,
        pin_memory=True,
        **options,
    )
    totals = torch.zeros(4, dtype=torch.float64, device=context.device)
    maximum_grad_norm = 0.0
    torch.cuda.synchronize(context.device)
    started = time.perf_counter()
    for batch_number, graph in enumerate(loader):
        graph = graph.to(context.device, non_blocking=True)
        assignment, graph_count = graph.batch, graph.num_graphs
        optimizer.zero_grad(set_to_none=True)
        prediction, mean, logvar = model(graph)
        squared = (
            ((prediction - graph.target) * data.std.to(context.device))
            .square()
            .mean(dim=1)
        )
        counts = torch.bincount(assignment, minlength=graph_count)
        per_graph = (
            torch.zeros(graph_count, device=context.device).scatter_add_(
                0, assignment, squared
            )
            / counts
        )
        reconstruction_sum = per_graph.sum()
        kl_sum = -0.5 * (1 + logvar - mean.square() - logvar.exp()).sum()
        denominators = context.sum(
            torch.tensor(
                [graph_count, mean.numel()],
                dtype=torch.float64,
                device=context.device,
            )
        )
        # DDP averages parameter gradients across ranks: multiply each local numerator by W.
        loss = context.world * (
            reconstruction_sum / denominators[0].float()
            + spec["recipe"]["kl_weight"] * kl_sum / denominators[1].float()
        )
        finite = context.sum(torch.isfinite(loss).to(torch.int64))
        if finite.item() != context.world:
            raise FloatingPointError(
                f"nonfinite loss at epoch {epoch}, batch {batch_number}"
            )
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), spec["recipe"]["clip_norm"], error_if_nonfinite=True
        )
        optimizer.step()
        maximum_grad_norm = max(maximum_grad_norm, float(norm))
        totals += torch.stack(
            (
                reconstruction_sum.detach().double(),
                torch.tensor(float(graph_count), device=context.device),
                kl_sum.detach().double(),
                torch.tensor(float(mean.numel()), device=context.device),
            )
        )
    torch.cuda.synchronize(context.device)
    elapsed = max(context.gather(time.perf_counter() - started))
    totals = context.sum(totals).cpu().tolist()
    maximum_grad_norm = max(context.gather(maximum_grad_norm))
    expected = 24 if acceptance else 1000
    if totals[1] != expected:
        raise ValueError(
            "epoch must consume every selected trajectory once, without padding/dropping"
        )
    return {
        "epoch": epoch,
        "sampled_frames": int(totals[1]),
        "optimizer_steps": len(loader),
        "train_sample_uv_mse": totals[0] / totals[1],
        "train_sample_kl": totals[2] / totals[3],
        "gradient_norm_max_before_clip": maximum_grad_norm,
        "train_seconds": elapsed,
        "training_gpu_seconds": context.world * elapsed,
        "frames_per_second": totals[1] / elapsed,
        "actual_global_tail": 8,
        "learning_rate": optimizer.param_groups[0]["lr"],
    }


def save_state(
    root, model, optimizer, state, metadata, context, persistent=None
) -> None:
    local = {
        "rng": capture_rng(context.device),
        "buffers": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.named_buffers()
        },
    }
    rank_states = context.gather(local)

    def save():
        payload = {
            "format": "vgae_cf.training_checkpoint.v1",
            "metadata": metadata,
            "model": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
            "optimizer": optimizer.state_dict(),
            "state": state,
            "rank_states": rank_states,
            "data_cursor": {
                "next_epoch": state["epoch"] + 1,
                "next_batch": 0,
                "frame_algorithm": "numpy.SeedSequence([seed, epoch, trajectory])",
            },
        }
        if persistent is not None:
            atomic_checkpoint(root / persistent, payload)
        atomic_checkpoint(root / "latest.pt", payload)
        if state["selection"]["best_checkpoint"] is not None:
            write_json(
                root / "best.json",
                {
                    "checkpoint": state["selection"]["best_checkpoint"],
                    "epoch": state["selection"]["best_epoch"],
                    "validation24_uv_mse": state["selection"]["best_mse"],
                    "evaluation_directory": state["selection"]["best_evaluation"],
                },
            )

    context.primary_call(save)


def export_codec(root: Path, model, metadata: dict, selection: dict) -> None:
    normalization = metadata["data"]["manifest"]["train_only_normalization"]
    codec_metadata = {
        "architecture": metadata["spec"]["architecture"],
        "normalization": {
            "field_mean": normalization["field_mean"][:2],
            "field_std": normalization["field_std"][:2],
            "inlet_mean": normalization["inlet_mean"],
            "inlet_std": normalization["inlet_std"],
        },
        "fields": ["u", "v"],
        "field_axes": ["fine_node", "UV"],
        "latent_axes": ["level_3_node", "channel"],
        "latent_standardized": False,
        "graph": {
            "coarse_level": 3,
            "precomputed_levels": 5,
            "coarsening": "Guillard",
            "relative_position_scales": [0.15, 0.3, 0.6, 1.2, 2.4],
            "pooling": "unchanged learned MeshDownMP sum aggregation",
            "positions": "graph.pos_3",
            "adjacency": "graph.edge_index_3",
            "node_order": "unchanged cached Guillard coarse-node order",
        },
        "conditions": [
            "normalized static inlet speed",
            "interior/outlet one-hot",
            "inlet one-hot",
            "wall one-hot",
        ],
        "edge_conditions": "2D relative edge position, scaled as upstream",
        "condition_encoder_depths": [1, 1, 1],
        "decoder_depths": list(reversed(metadata["spec"]["config"]["depths"])),
        "boundary": "inlet and wall UV from stored frame 0; no future boundary values",
        "checkpoint_selection": selection,
        "data_identity": metadata["data"],
        "source_commit": metadata["source"]["commit"],
        "test_accessed": False,
    }
    atomic_checkpoint(
        root / "codec.pt",
        {
            "format": "vgae_cf.codec.v1",
            "metadata": codec_metadata,
            "model": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
        },
    )
    write_json(root / "codec_metadata.json", codec_metadata)


def run(args) -> None:
    context = Context()
    root = None
    attempt = None
    started = time.perf_counter()
    locks = ExitStack()
    try:
        environment = context.verify_environment(args.environment)
        spec, plan = context.primary_call(lambda: slot(Path(args.campaign), args.slot))
        source = source_identity()
        if source["files"] != plan["source"]["files"]:
            raise ValueError(
                "source changed since campaign creation; preserve the registered source snapshot"
            )
        acceptance = args.mode == "acceptance"
        root = (
            Path(args.acceptance_root) / spec["id"]
            if acceptance
            else Path(args.campaign) / "runs" / spec["id"]
        )
        root = root.expanduser().resolve()
        context.primary_call(lambda: locks.enter_context(file_lock(root)) and None)
        if context.primary_call(lambda: (root / "result.json").exists()):
            if context.primary:
                print(
                    f"{spec['id']} is already complete; no training was launched",
                    flush=True,
                )
            return

        def check_restart():
            latest_exists = (root / "latest.pt").exists()
            old_attempts = list((root / "attempts").glob("attempt_*"))
            if args.resume and not latest_exists:
                raise ValueError(
                    "no rolling checkpoint; use --retry-initial for a failed initial attempt"
                )
            if latest_exists and (not args.resume or args.retry_initial):
                raise ValueError(
                    "existing rolling state requires --resume and forbids a fresh initialization"
                )
            if old_attempts and not latest_exists and not args.retry_initial:
                raise ValueError(
                    "initial attempt failed; --retry-initial reuses the slot and retains its failure records"
                )

        context.primary_call(check_restart)
        attempt = Path(
            context.primary_call(
                lambda: str(next_attempt(root / "attempts", "attempt"))
            )
        )
        context.primary_call(
            lambda: write_json(
                attempt / "started.json",
                {"time": utc_now(), "resume": args.resume, "mode": args.mode},
            )
        )
        data = Dataset(args.data)
        if not (data.directory / "graph_inventory.json").is_file():
            raise ValueError(
                "run prepare first to freeze and persist the shared graph hierarchy"
            )
        metadata = {
            "spec": spec,
            "mode": args.mode,
            "source": source,
            "data": data.identity(),
            "environment": environment,
        }

        def bind_execution_contract():
            campaign_root = Path(args.campaign)
            contract = {"environment": environment, "data": data.identity()}
            with file_lock(campaign_root):
                destination = campaign_root / "execution_contract.json"
                if destination.exists():
                    if read_json(destination) != contract:
                        raise ValueError(
                            "all campaign slots and acceptance must share the registered data/environment"
                        )
                else:
                    write_json(destination, contract)

        context.primary_call(bind_execution_contract)
        context.primary_call(lambda: write_json(attempt / "metadata.json", metadata))
        seed_everything(spec["seed"])
        model = build_model(spec["architecture"], context.device)
        distributed_model = DistributedDataParallel(
            model, device_ids=[context.local_rank], broadcast_buffers=True
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=spec["recipe"]["learning_rate"],
            betas=tuple(spec["recipe"]["betas"]),
            eps=spec["recipe"]["eps"],
            weight_decay=0,
        )
        seed_everything(spec["seed"] * 1000 + context.rank)
        state = {
            "epoch": 0,
            "phase": "training",
            "exposures": 0,
            "optimizer_steps": 0,
            "train_seconds": 0.0,
            "committed_wall_seconds": 0.0,
            "history": [],
            "peak_memory": [0] * 4,
            "selection": {
                "best_mse": None,
                "best_epoch": None,
                "best_checkpoint": None,
                "best_evaluation": None,
                "bad_evaluations": 0,
                "lr_bad_evaluations": 0,
                "valid_evaluations": 0,
                "learning_rate": spec["recipe"]["learning_rate"],
            },
        }
        if args.resume:
            saved = torch.load(
                root / "latest.pt", map_location="cpu", weights_only=True
            )
            for key in ("spec", "mode", "data", "environment"):
                if saved["metadata"][key] != metadata[key]:
                    raise ValueError(f"resume changed {key}")
            if saved["metadata"]["source"]["files"] != source["files"]:
                raise ValueError("resume changed executable source")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            state = saved["state"]
            if (
                saved["data_cursor"]["next_epoch"] != state["epoch"] + 1
                or saved["data_cursor"]["next_batch"] != 0
            ):
                raise ValueError("unsupported checkpoint data cursor")
            for name, buffer in model.named_buffers():
                buffer.copy_(
                    saved["rank_states"][context.rank]["buffers"][name].to(
                        context.device
                    )
                )
            restore_rng(saved["rank_states"][context.rank]["rng"], context.device)
            del saved
        wall_before = state["committed_wall_seconds"]
        torch.cuda.reset_peak_memory_stats(context.device)
        count = 4 if acceptance else 24
        train_indices = monitor_indices(data.splits["train"], count)
        validation_indices = monitor_indices(data.splits["validation"], count)
        maximum_epochs = 2 if acceptance else spec["recipe"]["maximum_epochs"]
        minimum_epochs = 1 if acceptance else spec["recipe"]["minimum_epochs"]
        for epoch in range(state["epoch"] + 1, maximum_epochs + 1):
            if state["phase"] != "training":
                break
            row = train_epoch(
                distributed_model,
                optimizer,
                data,
                spec,
                epoch,
                context,
                args.workers,
                acceptance,
            )
            state["epoch"] = epoch
            state["exposures"] += row["sampled_frames"]
            state["optimizer_steps"] += row["optimizer_steps"]
            state["train_seconds"] += row["train_seconds"]
            persistent = None
            if (
                epoch == 1
                or epoch % spec["recipe"]["validation_every"] == 0
                or epoch == maximum_epochs
            ):
                evaluation_root = Path(
                    context.primary_call(
                        lambda: str(
                            next_attempt(root / "evaluations", f"epoch_{epoch:05d}")
                        )
                    )
                )
                train_result = evaluate(
                    model, data, train_indices, context, evaluation_root / "train24"
                )
                validation = evaluate(
                    model,
                    data,
                    validation_indices,
                    context,
                    evaluation_root / "validation24",
                )
                state["selection"], improved = update_selection(
                    state["selection"],
                    validation["metrics"]["uv_mse"],
                    epoch,
                    spec["recipe"],
                )
                persistent = f"checkpoints/{evaluation_root.name}.pt"
                if improved:
                    state["selection"]["best_checkpoint"] = persistent
                    state["selection"]["best_evaluation"] = evaluation_root.relative_to(
                        root
                    ).as_posix()
                for group in optimizer.param_groups:
                    group["lr"] = state["selection"]["learning_rate"]
                row.update(
                    train24_uv_mse=train_result["metrics"]["uv_mse"],
                    validation24_uv_mse=validation["metrics"]["uv_mse"],
                    evaluation=evaluation_root.relative_to(root).as_posix(),
                    improved=improved,
                    next_learning_rate=state["selection"]["learning_rate"],
                )
            should_stop = (
                epoch >= minimum_epochs
                and state["selection"]["bad_evaluations"]
                >= spec["recipe"]["early_stop_patience"]
            )
            if epoch == maximum_epochs or should_stop:
                state["phase"] = "final_evaluation"
                state["stop_reason"] = (
                    "early_stopping" if should_stop else "maximum_epochs"
                )
            state["history"].append(row)
            current_peaks = context.gather(
                torch.cuda.max_memory_allocated(context.device)
            )
            state["peak_memory"] = [
                max(a, b) for a, b in zip(state["peak_memory"], current_peaks)
            ]
            state["committed_wall_seconds"] = (
                wall_before + time.perf_counter() - started
            )
            save_state(root, model, optimizer, state, metadata, context, persistent)
            context.primary_call(
                lambda: write_json(
                    root / "status.json",
                    {
                        "status": state["phase"],
                        "epoch": epoch,
                        "updated": utc_now(),
                        "best_validation24_uv_mse": state["selection"]["best_mse"],
                        "exposures": state["exposures"],
                    },
                )
            )
            if context.primary:
                print(
                    f"epoch={epoch} loss={row['train_sample_uv_mse']:.8g} best={state['selection']['best_mse']} lr={optimizer.param_groups[0]['lr']:.3g}",
                    flush=True,
                )
            if (
                args.stop_after_epoch is not None
                and epoch >= args.stop_after_epoch
                and state["phase"] == "training"
            ):
                context.primary_call(
                    lambda: write_json(
                        attempt / "exit.json",
                        {
                            "status": "paused_at_epoch_boundary",
                            "exit_code": 0,
                            "time": utc_now(),
                        },
                    )
                )
                return
        selection = state["selection"]
        selected = torch.load(
            root / selection["best_checkpoint"], map_location="cpu", weights_only=True
        )
        model.load_state_dict(selected["model"], strict=True)
        del selected
        final_root = Path(
            context.primary_call(
                lambda: str(next_attempt(root / "evaluations", "final_validation"))
            )
        )
        final_indices = validation_indices if acceptance else data.splits["validation"]
        final_result = evaluate(
            model, data, final_indices, context, final_root, save_fields=True
        )
        context.primary_call(lambda: final_figures(final_result, final_root))
        context.primary_call(lambda: export_codec(root, model, metadata, selection))
        final_peaks = context.gather(torch.cuda.max_memory_allocated(context.device))
        state["peak_memory"] = [
            max(a, b) for a, b in zip(state["peak_memory"], final_peaks)
        ]
        best_evaluation = root / selection["best_evaluation"]
        result = {
            "status": "completed",
            "mode": args.mode,
            "spec": spec,
            "best_epoch": selection["best_epoch"],
            "best_checkpoint": selection["best_checkpoint"],
            "training_epochs": state["epoch"],
            "stop_reason": state["stop_reason"],
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "train24": read_json(best_evaluation / "train24" / "summary.json"),
            "validation24": read_json(
                best_evaluation / "validation24" / "summary.json"
            ),
            "validation100"
            if not acceptance
            else "acceptance_validation": final_result,
            "final_evaluation_directory": final_root.relative_to(root).as_posix(),
            "training_exposures": state["exposures"],
            "optimizer_steps": state["optimizer_steps"],
            "training_gpu_seconds": state["train_seconds"] * 4,
            "committed_gpu_seconds": (wall_before + time.perf_counter() - started) * 4,
            "peak_allocated_bytes_by_rank": state["peak_memory"],
            "cost_scope": "completed epochs plus current attempt; crashed uncommitted work is retained in attempt/launcher logs",
            "history": state["history"],
            "environment": environment,
            "source_commit": source["commit"],
            "data_identity": data.identity(),
            "test_accessed": False,
            "completed_at": utc_now(),
        }
        context.primary_call(lambda: write_json(root / "result.json", result))
        context.primary_call(
            lambda: write_json(
                root / "status.json",
                {"status": "completed", "epoch": state["epoch"], "updated": utc_now()},
            )
        )
        context.primary_call(
            lambda: write_json(
                attempt / "exit.json",
                {"status": "completed", "exit_code": 0, "time": utc_now()},
            )
        )
    except BaseException:
        if attempt is not None:
            write_json(
                attempt / f"failure_rank{context.rank}.json",
                {
                    "time": utc_now(),
                    "traceback": traceback.format_exc(),
                    "attempt_wall_seconds": time.perf_counter() - started,
                    "exit_code": 1,
                },
            )
        if attempt is not None and context.primary:
            write_json(root / "status.json", {"status": "failed", "updated": utc_now()})
        raise
    finally:
        locks.close()
        context.close()

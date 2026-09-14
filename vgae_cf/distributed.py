"""Single-node, four-process CUDA/NCCL execution and reproducible rank state."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
from importlib.metadata import PackageNotFoundError, version
import os
import platform
import random
import subprocess
import sys
import traceback

import h5py
import numpy as np
import torch
import torch.distributed as dist

from .io import read_json


def capture_rng(device: torch.device) -> dict:
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
    }


def restore_rng(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    name, keys, position, gaussian, cached = state["numpy"]
    np.random.set_state(
        (name, np.asarray(keys, dtype=np.uint32), position, gaussian, cached)
    )
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


@contextmanager
def isolated_rng(device: torch.device):
    saved = capture_rng(device)
    try:
        yield
    finally:
        restore_rng(saved, device)


class Context:
    def __init__(self):
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        if self.world != 4 or int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != 4:
            raise RuntimeError("launch exactly four workers on one node with torchrun")
        if platform.system() != "Linux" or not torch.cuda.is_available():
            raise RuntimeError(
                "formal execution requires Linux and four NVIDIA CUDA GPUs"
            )
        if torch.cuda.device_count() != 4:
            raise RuntimeError(
                "expose exactly the four allocated GPUs through CUDA_VISIBLE_DEVICES"
            )
        self.device = torch.device("cuda", self.local_rank)
        torch.cuda.set_device(self.device)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))

    @property
    def primary(self) -> bool:
        return self.rank == 0

    def gather(self, value: object) -> list:
        values = [None] * self.world
        dist.all_gather_object(values, value)
        return values

    def broadcast(self, value: object) -> object:
        values = [value if self.primary else None]
        dist.broadcast_object_list(values, src=0)
        return values[0]

    def primary_call(self, function):
        result = None
        if self.primary:
            try:
                result = (True, function())
            except Exception:
                result = (False, traceback.format_exc())
        ok, payload = self.broadcast(result)
        if not ok:
            raise RuntimeError(payload)
        return payload

    def all_call(self, function):
        try:
            result = (True, function())
        except Exception:
            result = (False, traceback.format_exc())
        results = self.gather(result)
        failures = [
            f"rank {i}: {value}" for i, (ok, value) in enumerate(results) if not ok
        ]
        if failures:
            raise RuntimeError("\n".join(failures))
        return [value for _, value in results]

    def sum(self, tensor: torch.Tensor) -> torch.Tensor:
        value = tensor.detach().clone()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    def barrier(self) -> None:
        dist.barrier()

    def environment(self) -> dict:
        def package_version(name):
            try:
                return version(name)
            except PackageNotFoundError:
                return None

        properties = torch.cuda.get_device_properties(self.device)
        local = {
            "gpu": properties.name,
            "memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
        devices = self.gather(local)
        topology = self.primary_call(
            lambda: subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        driver = self.primary_call(
            lambda: (
                subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                .stdout.strip()
                .splitlines()
            )
        )
        return {
            "format": "vgae_cf.four_gpu_environment.v1",
            "os": platform.platform(),
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "nccl": list(torch.cuda.nccl.version())
            if isinstance(torch.cuda.nccl.version(), tuple)
            else torch.cuda.nccl.version(),
            "packages": {
                name: package_version(name)
                for name in (
                    "torchvision",
                    "torch-geometric",
                    "torch-scatter",
                    "torch-sparse",
                    "pyg-lib",
                    "numpy",
                    "h5py",
                    "matplotlib",
                )
            },
            "hdf5": h5py.version.hdf5_version,
            "world_size": 4,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
            "backend": "nccl",
            "devices": devices,
            "topology": topology,
            "driver_versions": driver,
            "precision": "fp32",
            "tf32": False,
            "cudnn_benchmark": False,
        }

    def verify_environment(self, expected_file) -> dict:
        actual = self.environment()
        expected = self.primary_call(lambda: read_json(expected_file))
        if actual != expected:
            raise RuntimeError(
                "execution environment differs from the recorded formal four-GPU profile"
            )
        return actual

    def close(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

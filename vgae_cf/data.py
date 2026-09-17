"""Pinned Train/Validation UVP fields and unchanged mesh coarsening."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch_geometric.utils import coalesce

from dgn4cfd.graph import Graph
from dgn4cfd.transforms import AddDirichletMask, MeshCoarsening, ScaleEdgeAttr
from .io import read_json, write_json


DATA_REPOSITORY = "dm-meshgraphnets/airfoil"
DATA_REVISION = "airfoil.uvp.stride8.first75.v1"
DATA_FILE = "airfoil_stride8_75frames.h5"
MANIFEST_FILE = "airfoil_stride8_75frames_manifest.json"
FORMAT = "dgn4cfd.mgn_airfoil_uvp_temporal_stride.v1"
GRAPH_FORMAT = "airfoil.uvp_graph_hierarchy.v1"


def monitor_indices(indices: tuple[int, ...], count: int = 24) -> tuple[int, ...]:
    positions = np.rint(
        np.linspace(0, len(indices) - 1, min(count, len(indices)))
    ).astype(int)
    return tuple(indices[int(position)] for position in positions)


def sampled_frame(seed: int, epoch: int, trajectory: int) -> int:
    generator = np.random.default_rng(np.random.SeedSequence([seed, epoch, trajectory]))
    return int(generator.integers(0, 75))


class Dataset:
    """Only the released Train/Validation trajectories can be read."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser().resolve(strict=True)
        self.dataset_file = self.directory / DATA_FILE
        self.manifest = read_json(self.directory / MANIFEST_FILE)
        m = self.manifest
        if m.get("format") != FORMAT or m.get("frames") != 75:
            raise ValueError("expected the released 75-frame stride-8 manifest")
        if m.get("temporal_stride") != 8 or not np.isclose(
            m.get("frame_dt", 0), 0.0016
        ):
            raise ValueError("expected stride=8 and dt=0.0016")
        if m.get("phase_offset", 0) != 0 or m.get("phase_augmentation", False):
            raise ValueError("only phase-zero data are supported")
        if m.get("source_frame_indices") != list(range(0, 600, 8)):
            raise ValueError("stored frames must correspond to raw 0,8,...,592")
        if m.get("test_accessed") not in (False, None):
            raise ValueError("the dataset must keep Test sealed")
        self.splits = {key: tuple(value) for key, value in m["splits"].items()}
        if self.splits != {
            "train": tuple(range(1000)),
            "validation": tuple(range(1000, 1100)),
        }:
            raise ValueError(
                "only the official Train 0..999 / Validation 1000..1099 split is supported"
            )
        self.normalization = copy.deepcopy(m["train_only_normalization"])
        self.mean = torch.tensor(self.normalization["field_mean"], dtype=torch.float32)
        self.std = torch.tensor(self.normalization["field_std"], dtype=torch.float32)
        if (
            self.mean.shape != (3,)
            or self.std.shape != (3,)
            or not torch.isfinite(self.mean).all()
        ):
            raise ValueError("invalid Train UVP normalization")
        if not torch.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("invalid Train UVP standard deviation")
        if float(self.normalization["inlet_std"]) <= 0:
            raise ValueError("invalid inlet normalization")
        self.graph_directory = self.directory / "uvp_graphs"
        self._graphs: dict[int, Graph] = {}
        with h5py.File(self.dataset_file, "r") as handle:
            required = {
                "format": FORMAT,
                "frames": 75,
                "temporal_stride": 8,
                "phase_offset": 0,
                "phase_augmentation": False,
                "test_accessed": False,
                "train_count": 1000,
                "validation_count": 100,
                "frame_dt": 0.0016,
            }
            if any(handle.attrs.get(key) != value for key, value in required.items()):
                raise ValueError(
                    "HDF5 metadata disagrees with the released data contract"
                )
            for index in self.splits["train"] + self.splits["validation"]:
                group = handle[f"trajectory_{index:04d}"]
                if group["uvp"].shape[0] != 75 or group["uvp"].shape[-1] != 3:
                    raise ValueError("stored fields must have shape [75,N,3]")
                if group["mesh_pos"].shape != (group["uvp"].shape[1], 2):
                    raise ValueError("mesh and field node counts disagree")
                if group.attrs.get("source_split") != (
                    "train" if index < 1000 else "validation"
                ):
                    raise ValueError("source split differs from the official manifest")

    def identity(self) -> dict:
        return {
            "dataset_repository": DATA_REPOSITORY,
            "dataset_revision": DATA_REVISION,
            "dataset_bytes": self.dataset_file.stat().st_size,
            "manifest": self.manifest,
            "model_fields": ["u", "v", "p"],
            "coarse_level": 3,
            "graph_format": GRAPH_FORMAT,
            "test_accessed": False,
        }

    def _check_index(self, index: int) -> None:
        if index not in self.splits["train"] and index not in self.splits["validation"]:
            raise ValueError("trajectory is outside Train/Validation")

    def graph_identity(self) -> dict:
        return {
            "dataset_revision": DATA_REVISION,
            "dataset_bytes": self.dataset_file.stat().st_size,
            "normalization": self.normalization,
            "format": GRAPH_FORMAT,
        }

    def read_uvp(self, index: int, start: int = 0, stop: int = 65) -> np.ndarray:
        self._check_index(index)
        if not 0 <= start < stop <= 75:
            raise ValueError("field access must remain within stored frames 0..74")
        with h5py.File(self.dataset_file, "r") as handle:
            values = np.asarray(
                handle[f"trajectory_{index:04d}/uvp"][start:stop, :, :],
                dtype=np.float32,
            )
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite UVP reference: trajectory {index}")
        return values

    def geometry(self, index: int) -> dict:
        self._check_index(index)
        with h5py.File(self.dataset_file, "r") as handle:
            group = handle[f"trajectory_{index:04d}"]
            return {
                "points": np.asarray(group["mesh_pos"], dtype=np.float32),
                "cells": np.asarray(group["cells"], dtype=np.int64),
                "node_type": np.asarray(group["node_type"], dtype=np.int64).reshape(-1),
                "inlet_velocity": float(group.attrs["inlet_velocity"]),
            }

    def static_graph(self, index: int) -> Graph:
        self._check_index(index)
        if index not in self._graphs:
            cache = self.graph_directory / f"{index:04d}.pt"
            if cache.is_file():
                saved = torch.load(cache, map_location="cpu", weights_only=True)
                if (
                    saved["format"] != GRAPH_FORMAT
                    or saved["data_identity"] != self.graph_identity()
                ):
                    raise ValueError(
                        "graph cache belongs to different data or preprocessing"
                    )
                graph = Graph.from_dict(saved["graph"])
            else:
                sample = self.geometry(index)
                points, cells, labels = (
                    sample["points"],
                    sample["cells"],
                    sample["node_type"],
                )
                if (
                    not np.isfinite(points).all()
                    or cells.ndim != 2
                    or cells.shape[1] != 3
                ):
                    raise ValueError("invalid triangular geometry")
                if cells.min() < 0 or cells.max() >= len(points):
                    raise ValueError("cell index is outside the mesh")
                if not set(np.unique(labels)).issubset({0, 2, 4}):
                    raise ValueError("unknown Airfoil node type")
                graph = Graph()
                graph.pos = torch.from_numpy(points)
                cells_tensor = torch.from_numpy(cells)
                source = cells_tensor[:, (0, 1, 2)].reshape(-1)
                target = cells_tensor[:, (1, 2, 0)].reshape(-1)
                edges = torch.stack((source, target))
                edges = edges[:, edges[0] != edges[1]]
                graph.edge_index = coalesce(
                    torch.cat((edges, edges.flip(0)), dim=1), sort_by_row=False
                )
                graph.edge_attr = (
                    graph.pos[graph.edge_index[1]] - graph.pos[graph.edge_index[0]]
                )
                graph.node_type = torch.from_numpy(labels)
                graph.bound = torch.zeros(len(points), dtype=torch.uint8)
                graph.bound[graph.node_type == 4] = 2
                graph.bound[graph.node_type == 5] = 3
                graph.bound[graph.node_type == 2] = 4
                graph.omega = torch.zeros(len(points), 3, dtype=torch.float32)
                graph.omega[(graph.node_type == 0) | (graph.node_type == 5), 0] = 1
                graph.omega[graph.node_type == 4, 1] = 1
                graph.omega[graph.node_type == 2, 2] = 1
                inlet = (
                    sample["inlet_velocity"] - self.normalization["inlet_mean"]
                ) / self.normalization["inlet_std"]
                graph.glob = torch.full(
                    (len(points), 1), float(inlet), dtype=torch.float32
                )
                graph = ScaleEdgeAttr(0.15)(graph)
                graph = AddDirichletMask(3, [0, 1], [])(graph)
                graph = MeshCoarsening(
                    num_scales=5,
                    rel_pos_scaling=[0.15, 0.3, 0.6, 1.2, 2.4],
                    scalar_rel_pos=True,
                )(graph)
                initial = torch.from_numpy(self.read_uvp(index, 0, 1)[0])
                graph.boundary_values = torch.where(
                    graph.dirichlet_mask,
                    (initial - self.mean) / self.std,
                    torch.zeros_like(initial),
                )
            if len(self._graphs) >= 32:
                self._graphs.pop(next(iter(self._graphs)))
            self._graphs[index] = graph
        # Upstream Collater offsets tensors in place. Never hand it a shared cache.
        return self._graphs[index].clone()

    def frame(self, index: int, frame: int) -> Graph:
        graph = self.static_graph(index)
        field = torch.from_numpy(self.read_uvp(index, frame, frame + 1)[0])
        graph.target = (field - self.mean) / self.std
        graph.field = graph.target.clone()
        graph.trajectory_id = torch.tensor([index], dtype=torch.long)
        graph.frame_id = torch.tensor([frame], dtype=torch.long)
        return graph


class EpochFrames(torch.utils.data.Dataset):
    def __init__(self, data: Dataset, seed: int, epoch: int):
        self.data, self.seed, self.epoch = data, seed, epoch
        self.indices = data.splits["train"]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, ordinal: int) -> Graph:
        index = self.indices[ordinal]
        return self.data.frame(index, sampled_frame(self.seed, self.epoch, index))


def prepare(directory: Path, *, download_only: bool = False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in (DATA_FILE, MANIFEST_FILE):
        if not (directory / name).is_file():
            raise FileNotFoundError("run python -m airfoil_data.prepare first: " + name)
    data = Dataset(directory)
    write_json(directory / "data_identity.json", data.identity())
    if download_only:
        return
    data.graph_directory.mkdir(parents=True, exist_ok=True)
    inventory = []
    for index in data.splits["train"] + data.splits["validation"]:
        graph = data.static_graph(index)
        destination = data.graph_directory / f"{index:04d}.pt"
        if not destination.exists():
            temporary = destination.with_suffix(".tmp")
            torch.save(
                {
                    "format": GRAPH_FORMAT,
                    "data_identity": data.graph_identity(),
                    "graph": graph.to_dict(),
                },
                temporary,
            )
            os.replace(temporary, destination)
        inventory.append(
            {
                "trajectory": index,
                "nodes": graph.num_nodes,
                "latent_nodes": graph.pos_3.shape[0],
            }
        )
        data._graphs.pop(index, None)
        if len(inventory) % 50 == 0:
            print(f"Prepared {len(inventory)}/1100 static graphs", flush=True)
    write_json(directory / "graph_inventory.json", inventory)

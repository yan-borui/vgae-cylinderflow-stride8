"""UV VGAE and the representation interface used by a subsequent DiT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from dgn4cfd.graph import Graph
from dgn4cfd.nn.models import VGAE


@dataclass(frozen=True)
class DecoderContext:
    nodes: tuple
    edges: tuple
    edge_indices: tuple
    batches: tuple


@dataclass(frozen=True)
class Posterior:
    sample: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor
    context: DecoderContext


class UVVGAE(VGAE):
    def encode_latent(self, graph: Graph, normalized_uv: torch.Tensor) -> Posterior:
        z, mean, logvar, nodes, edges, indices, batches = super().encode(
            graph, normalized_uv
        )
        return Posterior(
            z,
            mean,
            logvar,
            DecoderContext(
                tuple(nodes),
                tuple(edges),
                tuple(indices),
                tuple(batches),
            ),
        )

    def conditions(self, graph: Graph) -> DecoderContext:
        """Encode static inlet, node type and mesh; no target field is read."""
        values = self.cond_encoder(
            graph,
            torch.cat((graph.glob, graph.omega), dim=1),
            graph.edge_attr,
            graph.edge_index,
        )
        return DecoderContext(*(tuple(item) for item in values))

    def decode_latent(
        self,
        graph: Graph,
        latent: torch.Tensor,
        context: DecoderContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latent.shape != (graph.pos_3.shape[0], self.latent_node_features):
            raise ValueError(
                "latent must have shape [number of level-3 nodes, latent channels]"
            )
        graph.edge_index, graph.batch = context.edge_indices[-1], context.batches[-1]
        raw = super().decode(
            graph,
            latent,
            list(context.nodes),
            list(context.edges),
            list(context.edge_indices),
            list(context.batches),
        )
        prediction = torch.where(graph.dirichlet_mask, graph.boundary_values, raw)
        return prediction, raw

    def forward(self, graph: Graph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior = self.encode_latent(graph, graph.field)
        prediction, _ = self.decode_latent(graph, posterior.sample, posterior.context)
        return prediction, posterior.mean, posterior.logvar


def build_model(architecture: dict, device: torch.device) -> UVVGAE:
    model = UVVGAE(arch=architecture, device=device)
    return nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)


class UVCodec:
    """Eval-only physical UV <-> unstandardized level-3 latent interface."""

    def __init__(self, artifact: str | Path, device: str | torch.device = "cuda"):
        payload = torch.load(artifact, map_location="cpu", weights_only=True)
        if payload.get("format") != "vgae_cf.codec.v1":
            raise ValueError("expected an exported UV codec")
        self.metadata = payload["metadata"]
        self.device = torch.device(device)
        self.model = build_model(self.metadata["architecture"], self.device)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.eval()
        self.mean = torch.tensor(
            self.metadata["normalization"]["field_mean"],
            device=self.device,
            dtype=torch.float32,
        )
        self.std = torch.tensor(
            self.metadata["normalization"]["field_std"],
            device=self.device,
            dtype=torch.float32,
        )

    @torch.inference_mode()
    def encode(self, graph: Graph, physical_uv: torch.Tensor) -> Posterior:
        """Return posterior mean/logvar/sample and reusable static decoder context."""
        graph = graph.clone().to(self.device)
        values = physical_uv.to(device=self.device, dtype=torch.float32)
        if values.shape != (graph.pos.shape[0], 2):
            raise ValueError("physical_uv must have shape [fine nodes, 2]")
        return self.model.encode_latent(graph, (values - self.mean) / self.std)

    @torch.inference_mode()
    def decode(
        self,
        graph: Graph,
        latent: torch.Tensor,
        context: DecoderContext | None = None,
    ) -> torch.Tensor:
        """Decode predicted or posterior-mean latents to physical UV."""
        graph = graph.clone().to(self.device)
        if context is None:
            context = self.model.conditions(graph)
        prediction, _ = self.model.decode_latent(
            graph, latent.to(device=self.device, dtype=torch.float32), context
        )
        return prediction * self.std + self.mean

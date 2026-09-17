"""UVP VGAE and the representation interface used by a subsequent DiT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from dgn4cfd.graph import Graph
from dgn4cfd.nn.blocks import InteractionNetwork
from dgn4cfd.nn.models import VGAE
from . import CODEC_FORMAT, PROTOCOL


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


class UVPVGAE(VGAE):
    def encode_latent(self, graph: Graph, normalized_uvp: torch.Tensor) -> Posterior:
        if normalized_uvp.shape != (graph.pos.shape[0], 3):
            raise ValueError("normalized_uvp must have shape [fine nodes, 3]")
        z, mean, logvar, nodes, edges, indices, batches = super().encode(
            graph, normalized_uvp
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
        if (
            graph.dirichlet_mask.shape != (graph.pos.shape[0], 3)
            or graph.dirichlet_mask[:, 2].any()
            or graph.boundary_values.shape != graph.dirichlet_mask.shape
        ):
            raise ValueError(
                "UVP decoding requires a three-channel mask with free pressure"
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


def build_model(
    architecture: dict,
    device: torch.device,
    *,
    activation_checkpointing: bool = False,
) -> UVPVGAE:
    if architecture.get("in_node_features") != 3:
        raise ValueError("the personal UVP VGAE requires three input/output fields")
    model = UVPVGAE(arch=architecture, device=device)
    for module in model.modules():
        if isinstance(module, InteractionNetwork):
            module.activation_checkpointing = activation_checkpointing
    # Match global-batch normalization across the four shards.
    return torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)


class UVPCodec:
    """Eval-only physical UVP <-> unstandardized level-3 latent interface."""

    def __init__(self, artifact: str | Path, device: str | torch.device = "cuda"):
        payload = torch.load(artifact, map_location="cpu", weights_only=True)
        if payload.get("format") != CODEC_FORMAT:
            raise ValueError(
                "expected an exported UVP codec; old UV codecs are incompatible"
            )
        self.metadata = payload["metadata"]
        if self.metadata.get("protocol") != PROTOCOL or self.metadata.get("fields") != [
            "u",
            "v",
            "p",
        ]:
            raise ValueError(
                "codec metadata does not describe the current UVP protocol"
            )
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
        if (
            self.mean.shape != (3,)
            or self.std.shape != (3,)
            or not torch.isfinite(self.mean).all()
            or not torch.isfinite(self.std).all()
            or (self.std <= 0).any()
        ):
            raise ValueError(
                "codec requires finite Train UVP normalization and positive std"
            )

    @torch.inference_mode()
    def encode(self, graph: Graph, physical_uvp: torch.Tensor) -> Posterior:
        """Return posterior mean/logvar/sample and reusable static decoder context."""
        graph = graph.clone().to(self.device)
        values = physical_uvp.to(device=self.device, dtype=torch.float32)
        if values.shape != (graph.pos.shape[0], 3):
            raise ValueError("physical_uvp must have shape [fine nodes, 3]")
        return self.model.encode_latent(graph, (values - self.mean) / self.std)

    @torch.inference_mode()
    def decode(
        self,
        graph: Graph,
        latent: torch.Tensor,
        context: DecoderContext | None = None,
    ) -> torch.Tensor:
        """Decode predicted or posterior-mean latents to physical UVP."""
        graph = graph.clone().to(self.device)
        if context is None:
            context = self.model.conditions(graph)
        prediction, _ = self.model.decode_latent(
            graph, latent.to(device=self.device, dtype=torch.float32), context
        )
        return prediction * self.std + self.mean

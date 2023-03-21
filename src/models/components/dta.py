from __future__ import annotations

import torch
import torch.nn as nn

from src.models.components.encoder import Encoder

__all__ = [
    "DeepDTA",
]


class DeepDTA(nn.Module):
    def __init__(
        self,
        protein_encoder: Encoder,
        ligand_encoder: Encoder,
        dense: nn.Module,
    ):
        super().__init__()

        self._ligand_encoder = ligand_encoder
        self._protein_encoder = protein_encoder
        self._dense = dense

    def forward(self, x_proteins: torch.Tensor, x_ligands: torch.Tensor) -> torch.Tensor:
        protein_embedding = self._protein_encoder.forward(x_proteins)
        ligand_embedding = self._ligand_encoder.forward(x_ligands)

        x = torch.cat((protein_embedding, ligand_embedding), dim=-1)
        x = self._dense.forward(x)
        return x

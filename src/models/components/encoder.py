from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["Encoder"]


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 100,
        filters: int = 32,
        kernel_size: tuple[int, ...] = (4,),
        embedding_size: int = 128,
    ):
        super().__init__()

        self._embedding = nn.Embedding(vocab_size, embedding_size)

        layers = []
        for i, size in enumerate(kernel_size, start=1):
            layers.append(nn.LazyConv1d(out_channels=filters * i, kernel_size=size, stride=1))
        layers.append(nn.MaxPool1d(kernel_size[-1]))
        layers.append(nn.Flatten(start_dim=1))
        self._conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._embedding.forward(x)
        x = self._conv.forward(x)
        return x

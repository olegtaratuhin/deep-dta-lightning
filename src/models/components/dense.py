from __future__ import annotations

from typing import Any, Protocol

import torch
import torch.nn as nn

__all__ = [
    "SimpleDenseNet",
]


class Initializer(Protocol):
    def __call__(self, tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError()


class SimpleDenseNet(nn.Module):
    def __init__(
        self,
        inner_sizes: list[int],
        output_size: int = 1,
        dropout: float = 0.1,
        output_init: Initializer | None = None,
    ):
        super().__init__()

        # # Fully connected
        # FC1 = Dense(1024, activation='relu')(encode_interaction)
        # FC2 = Dropout(0.1)(FC1)
        # FC2 = Dense(1024, activation='relu')(FC2)
        # FC2 = Dropout(0.1)(FC2)
        # FC2 = Dense(512, activation='relu')(FC2)

        layers = []
        for prev_size, next_size in zip(inner_sizes[:-1], inner_sizes[1:]):
            layers.append(nn.Linear(prev_size, next_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))

        # pop last dropout
        layers.pop()

        output_layer = nn.Linear(inner_sizes[-1], output_size)
        layers.append(output_layer)
        if output_init is not None:
            output_init(output_layer.weight)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.forward(x)


class LazyDenseNet(nn.Module):
    def __init__(
        self,
        inner_sizes: list[int],
        output_size: int = 1,
        dropout: float = 0.1,
        output_init: Initializer | None = None,
    ):
        super().__init__()

        # # Fully connected
        # FC1 = Dense(1024, activation='relu')(encode_interaction)
        # FC2 = Dropout(0.1)(FC1)
        # FC2 = Dense(1024, activation='relu')(FC2)
        # FC2 = Dropout(0.1)(FC2)
        # FC2 = Dense(512, activation='relu')(FC2)

        layers = []
        for size in inner_sizes:
            layers.append(nn.LazyLinear(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))

        # pop last dropout
        layers.pop()

        output_layer = nn.Linear(inner_sizes[-1], output_size)
        layers.append(output_layer)
        if output_init is not None:
            output_init(output_layer.weight)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.forward(x)


if __name__ == "__main__":
    # noinspection PyTypeChecker
    net = LazyDenseNet(
        inner_sizes=[1024, 1024, 512],
        output_size=1,
        dropout=0.1,
        output_init=nn.init.xavier_normal_,
    )
    print(net)

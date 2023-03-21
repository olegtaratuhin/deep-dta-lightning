from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["Encoder"]


class Encoder(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        filters: int = 32,
        kernel_size: tuple[int, ...] = (4,),
        embedding_size: int = 128,
    ):
        super().__init__()

        # XTinput = Input(shape=(FLAGS.max_seq_len, FLAGS.charseqset_size))
        self._embedding = nn.Embedding(max_seq_len, embedding_size)

        # ================================================================================
        # @ Convolution for smiles

        # encode_smiles = Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
        #                        strides=1)(encode_smiles)
        # encode_smiles = Conv1D(filters=NUM_FILTERS * 2, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
        #                        strides=1)(encode_smiles)
        # encode_smiles = Conv1D(filters=NUM_FILTERS * 3, kernel_size=FILTER_LENGTH1, activation='relu', padding='valid',
        #                        strides=1)(encode_smiles)
        # encode_smiles = GlobalMaxPooling1D()(encode_smiles)

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

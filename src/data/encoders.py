from __future__ import annotations

from typing import NewType

import numpy as np
import numpy.typing as npt

SMILES_ALPHABET = "#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTUVWYZ[\\]abcdefghilmnorstuy"
PROTEIN_ALPHABET = "ABCDEFGHIKLMNOPQRSTUVWXYZ"


class LabelEncoder:
    def __init__(
        self,
        alphabet: str,
        n_dim: int,
        missing_value: int | None = None,
    ):
        self._table = dict(zip(sorted(alphabet), range(len(alphabet))))
        self._n_dim = n_dim
        self._missing_value = len(self._table) + 1 if missing_value is None else missing_value

    def transform(self, data: str) -> npt.NDArray[int]:
        encoded = np.zeros(self._n_dim)
        for i, ch in enumerate(data[: self._n_dim]):
            encoded[i] = self._table.get(ch, -1)
        return encoded

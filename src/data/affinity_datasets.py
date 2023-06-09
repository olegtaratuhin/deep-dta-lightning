from __future__ import annotations

import json
import pathlib
import shutil
from typing import Callable, NewType

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch.utils.data

from src.data import encoders

__all__ = [
    "AffinityDataset",
]

AffinityMatrix = NewType("AffinityMatrix", npt.NDArray[float])
AffinityNormalizer = Callable[[AffinityMatrix], AffinityMatrix]
LigandEmbedding = NewType("LigandEmbedding", npt.NDArray[int])
ProteinEmbedding = NewType("ProteinEmbedding", npt.NDArray[int])


def _extract_interactions(
    proteins: npt.ArrayLike[ProteinEmbedding],
    ligands: npt.ArrayLike[LigandEmbedding],
    affinity: AffinityMatrix,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = ~(np.isnan(affinity) | np.isinf(affinity))
    ligand_idx, protein_idx = np.where(mask)
    return (
        torch.from_numpy(np.take_along_axis(proteins, protein_idx.reshape(-1, 1), axis=0)),
        torch.from_numpy(np.take_along_axis(ligands, ligand_idx.reshape(-1, 1), axis=0)),
        torch.from_numpy(affinity[mask].reshape(-1, 1)),
    )


def _load_ligands(path: pathlib.Path, encoder: encoders.LabelEncoder) -> npt.NDArray[int]:
    with (path / "ligands.json").open("r") as ligands_file:
        ligands: dict[str, str] = json.load(ligands_file)
    return np.array([encoder.transform(x) for x in ligands.values()], dtype=int)


def _load_proteins(path: pathlib.Path, encoder: encoders.LabelEncoder) -> npt.NDArray[int]:
    with (path / "proteins.json").open("r") as proteins_file:
        proteins: dict[str, str] = json.load(proteins_file)
    return np.array([encoder.transform(x) for x in proteins.values()], dtype=int)


def _load_affinity(
    path: pathlib.Path,
    log_scale: bool = False,
    scaler: AffinityNormalizer | None = None,
) -> AffinityMatrix:
    affinity_scores = np.load(str(path / "affinity.npy"))
    if log_scale:
        affinity_scores = -np.log10(affinity_scores / (np.power(10, 9)))
    if scaler is not None:
        # attention: this normalizer is applied to whole dataset, this might result in data leak
        affinity_scores = scaler(affinity_scores)
    return affinity_scores.astype(dtype="float32")  # type: ignore


class AffinityDataset(torch.utils.data.TensorDataset):
    """Affinity dataset between ligands and proteins."""

    def __init__(
        self,
        path: pathlib.Path | None = None,
        ligand_dim: int = 64,
        protein_dim: int = 512,
        log_scale: bool = False,
        scaler: AffinityNormalizer | None = None,
        threshold: float | None = None,
    ):
        self._ligand_encoder = encoders.LabelEncoder(
            alphabet=encoders.SMILES_ALPHABET,
            n_dim=ligand_dim,
        )
        self._protein_encoder = encoders.LabelEncoder(
            alphabet=encoders.PROTEIN_ALPHABET,
            n_dim=protein_dim,
        )

        # binding affinity datasets often benchmark using AUPR, this threshold is dataset-specific
        # and will be accessed by the model's configuration
        self._threshold = threshold

        if path is None:
            raise ValueError("Expecting a non-None path to load interaction from")
        proteins, ligands, affinity = _extract_interactions(
            proteins=_load_proteins(path, self._protein_encoder),
            ligands=_load_ligands(path, self._ligand_encoder),
            affinity=_load_affinity(path, log_scale, scaler),
        )
        super().__init__(proteins, ligands, affinity)


def _convert_from_original_format(
    original_path: pathlib.Path,
    converted_path: pathlib.Path,
) -> None:
    """Helper function to convert original DeepDTA dataset formats into this one."""
    shutil.copyfile(original_path / "ligands_iso.txt", converted_path / "ligands.json")
    shutil.copyfile(original_path / "proteins.txt", converted_path / "proteins.json")
    affinity_scores = np.load(str(original_path / "Y"), encoding="latin1", allow_pickle=True)
    np.save(str(converted_path / "affinity.npy"), affinity_scores)


class TabularAffinityDataset(torch.utils.data.TensorDataset):
    """Affinity dataset between ligands and proteins in pd.DataFrame.

    This class expects that the dataframe would have 3 columns:
      - "Drug" - ligands in smiles format
      - "Target" - proteins in canonical encoding
      - "Y" - binding affinity between a drug and a target
    """

    def __init__(
        self,
        df: pd.DataFrame,
        ligand_dim: int = 64,
        protein_dim: int = 512,
        threshold: float | None = None,
    ):
        self._ligand_encoder = encoders.LabelEncoder(
            alphabet=encoders.SMILES_ALPHABET,
            n_dim=ligand_dim,
        )
        self._protein_encoder = encoders.LabelEncoder(
            alphabet=encoders.PROTEIN_ALPHABET,
            n_dim=protein_dim,
        )

        # binding affinity datasets often benchmark using AUPR, this threshold is dataset-specific
        # and will be accessed by the model's configuration
        self._threshold = threshold

        encoded_proteins = df["Target"].apply(self._protein_encoder.transform)
        proteins = torch.tensor(np.vstack(encoded_proteins), dtype=torch.int)

        encoded_ligands = df["Drug"].apply(self._ligand_encoder.transform)
        ligands = torch.tensor(np.vstack(encoded_ligands), dtype=torch.int)

        affinity = torch.tensor(df[["Y"]].values, dtype=torch.float)
        super().__init__(proteins, ligands, affinity)

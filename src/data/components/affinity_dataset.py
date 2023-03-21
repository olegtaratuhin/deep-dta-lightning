from __future__ import annotations

import json
import math
import pathlib
import shutil
from typing import Literal, NewType

import numpy as np
import numpy.typing as npt
import torch.utils.data

from src.data.components.ligands import smiles_encoding_table
from src.data.components.proteins import protein_encoding_table

__all__ = [
    "AffinityDataset",
]


AffinityMatrix = NewType("AffinityMatrix", npt.NDArray[float])
LigandEmbedding = NewType("LigandEmbedding", npt.NDArray[int])
ProteinEmbedding = NewType("ProteinEmbedding", npt.NDArray[int])
EncodingModes = Literal["one-hot", "label"]


class LabelEncoder:
    def __init__(self, table: dict[str, int], n_dim: int | None = None):
        self._table = table
        self._n_dim = n_dim if n_dim is not None else len(table)

    def transform(self, data: str) -> npt.NDArray[int]:
        encoded = np.zeros(self._n_dim)
        for i, ch in enumerate(data[: self._n_dim]):
            encoded[i] = self._table[ch]
        return encoded


def _normalize(vector: npt.NDArray[float], base: float = math.e) -> npt.NDArray[float]:
    return -np.log10(vector / (np.power(base, 9)))


def _extract_interactions(
    proteins: npt.ArrayLike[ProteinEmbedding],
    ligands: npt.ArrayLike[LigandEmbedding],
    affinity: AffinityMatrix,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = ~np.isnan(affinity)
    ligand_idx, protein_idx = np.where(mask)
    return (
        torch.from_numpy(np.take_along_axis(proteins, protein_idx.reshape(-1, 1), axis=0)),
        torch.from_numpy(np.take_along_axis(ligands, ligand_idx.reshape(-1, 1), axis=0)),
        torch.from_numpy(affinity[mask].reshape(-1, 1)),
    )


def _load_ligands(path: pathlib.Path, encoder: LabelEncoder) -> npt.NDArray[int]:
    with (path / "ligands.json").open("r") as ligands_file:
        ligands: dict[str, str] = json.load(ligands_file)
    return np.array([encoder.transform(x) for x in ligands.values()], dtype=int)


def _load_proteins(path: pathlib.Path, encoder: LabelEncoder) -> npt.NDArray[int]:
    with (path / "proteins.json").open("r") as proteins_file:
        proteins: dict[str, str] = json.load(proteins_file)
    return np.array([encoder.transform(x) for x in proteins.values()], dtype=int)


def _load_affinity(path: pathlib.Path, normalize_value: float | None = None) -> AffinityMatrix:
    affinity_scores = np.load(str(path / "affinity.npy"))
    if normalize_value is not None:
        affinity_scores = _normalize(affinity_scores, base=normalize_value)
    return affinity_scores.astype(dtype=np.dtype("float32"))  # type: ignore


class AffinityDataset(torch.utils.data.TensorDataset):
    """Affinity dataset between ligands and proteins."""

    def __init__(
        self,
        path: pathlib.Path | None = None,
        normalize_value: float | None = None,
        ligand_dim: int = 64,
        protein_dim: int = 512,
    ):
        self._ligand_encoder = LabelEncoder(table=smiles_encoding_table(), n_dim=ligand_dim)
        self._protein_encoder = LabelEncoder(table=protein_encoding_table(), n_dim=protein_dim)

        if path is None:
            raise ValueError("Expecting a non-None path to load interaction from")
        proteins, ligands, affinity = _extract_interactions(
            proteins=_load_proteins(path, self._protein_encoder),
            ligands=_load_ligands(path, self._ligand_encoder),
            affinity=_load_affinity(path, normalize_value),
        )
        super().__init__(proteins, ligands, affinity)

    def encode_protein(self, protein: str) -> npt.NDArray[int]:
        return self._protein_encoder.transform(data=protein)

    def encode_ligand(self, ligand: str) -> npt.NDArray[int]:
        return self._ligand_encoder.transform(data=ligand)


def _convert_from_original_format(
    original_path: pathlib.Path,
    converted_path: pathlib.Path,
) -> None:
    """Helper function to convert original DeepDTA dataset formats into this one."""
    shutil.copyfile(original_path / "ligands_iso.txt", converted_path / "ligands.json")
    shutil.copyfile(original_path / "proteins.txt", converted_path / "proteins.json")
    affinity_scores = np.load(str(original_path / "Y"), encoding="latin1", allow_pickle=True)
    np.save(str(converted_path / "affinity.npy"), affinity_scores)


if __name__ == "__main__":
    DATA_PATH = "/Users/oleg.taratukhin/Code/quantori/deep-dta-lightning/data/davis"
    dataset = AffinityDataset(path=pathlib.Path(DATA_PATH), normalize_value=10.0)

    protein = "MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKKTQPKARLTDPIPTTETSIAPRQRPKAGQTQPNPGILPIQPALTPRKRATVQPPPQAAGSSNQPGLLASVPQPKPQAPPSQPLPQTQAKQPQAPPTPQQTPSTQAQGLPAQAQATPQHQQQLFLKQQQQQQQPPPAQQQPAGTFYQQQQAQTQQFQAVHPATQKPAIAQFPVVSQGGSQQQLMQNFYQQQQQQQQQQQQQQLATALHQQQLMTQQAALQQKPTMAAGQQPQPQPAAAPQPAPAQEPAIQAPVRQQPKVQTTPPPAVQGQKVGSLTPPSSPKTQRAGHRRILSDVTHSAVFGVPASKSTQLLQAAAAEASLNKSKSATTTPSGSPRTSQQNVYNPSEGSTWNPFDDDNFSKLTAEELLNKDFAKLGEGKHPEKLGGSAESLIPGFQSTQGDAFATTSFSAGTAEKRKGGQTVDSGLPLLSVSDPFIPLQVPDAPEKLIEGLKSPDTSLLLPDLLPMTDPFGSTSDAVIEKADVAVESLIPGLEPPVPQRLPSQTESVTSNRTDSLTGEDSLLDCSLLSNPTTDLLEEFAPTAISAPVHKAAEDSNLISGFDVPEGSDKVAEDEFDPIPVLITKNPQGGHSRNSSGSSESSLPNLARSLLLVDQLIDL"
    print("Protein encoding:")
    print(f"{protein=} -> \n{dataset.encode_protein(protein)}")

    ligand = "CC1=C2C=C(C=CC2=NN1)C3=CC(=CN=C3)OCC(CC4=CC=CC=C4)N"
    print("Ligand encoding:")
    print(f"{ligand=} -> \n{dataset.encode_ligand(ligand)}")

from __future__ import annotations

import json
import math
import pathlib
import shutil
from typing import Literal, NewType, Protocol

import numpy as np
import numpy.typing as npt
import torch.utils.data

_protein_table = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}
_can_smiles_table = {
    "#": 1,
    "%": 2,
    ")": 3,
    "(": 4,
    "+": 5,
    "-": 6,
    ".": 7,
    "1": 8,
    "0": 9,
    "3": 10,
    "2": 11,
    "5": 12,
    "4": 13,
    "7": 14,
    "6": 15,
    "9": 16,
    "8": 17,
    "=": 18,
    "A": 19,
    "C": 20,
    "B": 21,
    "E": 22,
    "D": 23,
    "G": 24,
    "F": 25,
    "I": 26,
    "H": 27,
    "K": 28,
    "M": 29,
    "L": 30,
    "O": 31,
    "N": 32,
    "P": 33,
    "S": 34,
    "R": 35,
    "U": 36,
    "T": 37,
    "W": 38,
    "V": 39,
    "Y": 40,
    "[": 41,
    "Z": 42,
    "]": 43,
    "_": 44,
    "a": 45,
    "c": 46,
    "b": 47,
    "e": 48,
    "d": 49,
    "g": 50,
    "f": 51,
    "i": 52,
    "h": 53,
    "m": 54,
    "l": 55,
    "o": 56,
    "n": 57,
    "s": 58,
    "r": 59,
    "u": 60,
    "t": 61,
    "y": 62,
}
_iso_smiles_table = {
    "#": 29,
    "%": 30,
    ")": 31,
    "(": 1,
    "+": 32,
    "-": 33,
    "/": 34,
    ".": 2,
    "1": 35,
    "0": 3,
    "3": 36,
    "2": 4,
    "5": 37,
    "4": 5,
    "7": 38,
    "6": 6,
    "9": 39,
    "8": 7,
    "=": 40,
    "A": 41,
    "@": 8,
    "C": 42,
    "B": 9,
    "E": 43,
    "D": 10,
    "G": 44,
    "F": 11,
    "I": 45,
    "H": 12,
    "K": 46,
    "M": 47,
    "L": 13,
    "O": 48,
    "N": 14,
    "P": 15,
    "S": 49,
    "R": 16,
    "U": 50,
    "T": 17,
    "W": 51,
    "V": 18,
    "Y": 52,
    "[": 53,
    "Z": 19,
    "]": 54,
    "\\": 20,
    "a": 55,
    "c": 56,
    "b": 21,
    "e": 57,
    "d": 22,
    "g": 58,
    "f": 23,
    "i": 59,
    "h": 24,
    "m": 60,
    "l": 25,
    "o": 61,
    "n": 26,
    "s": 62,
    "r": 27,
    "u": 63,
    "t": 28,
    "y": 64,
}

AffinityMatrix = NewType("AffinityMatrix", npt.NDArray[float])
LigandEmbedding = NewType("LigandEmbedding", npt.NDArray[int])
ProteinEmbedding = NewType("ProteinEmbedding", npt.NDArray[int])
EncodingModes = Literal["one-hot", "label"]


__all__ = [
    "AffinityDataset",
    "Encoder",
    "LabelEncoder",
]


class Encoder(Protocol):
    def transform(self, data: str) -> npt.NDArray[int]:
        raise NotImplementedError()


class OneHotEncoder(Encoder):
    def __init__(self, table: dict[str, int], n_dim: int | None = None):
        self._table = table
        self._n_dim = n_dim if n_dim is not None else len(table)

    def transform(self, data: str) -> npt.NDArray[int]:
        encoded = np.zeros((self._n_dim, len(self._table)))
        for i, ch in enumerate(data[: self._n_dim]):
            encoded[i, (self._table[ch] - 1)] = 1
        return encoded


class LabelEncoder(Encoder):
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


def _encoder_from_mode(
    mode: EncodingModes,
    table: dict[str, int],
    n_dim: int | None = None,
) -> Encoder:
    if mode not in ("one-hot", "label"):
        raise ValueError(f"Incorrect ligand encoding mode: {mode}")
    return (
        OneHotEncoder(n_dim=n_dim, table=table)
        if mode == "one-hot"
        else LabelEncoder(n_dim=n_dim, table=table)
    )


def _extract_interactions(
    proteins: npt.ArrayLike[ProteinEmbedding],
    ligands: npt.ArrayLike[LigandEmbedding],
    affinity: AffinityMatrix,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ligand_idx, protein_idx = np.where(~np.isnan(affinity))
    return (
        torch.from_numpy(np.take_along_axis(proteins, protein_idx.reshape(-1, 1), axis=0)),
        torch.from_numpy(np.take_along_axis(ligands, ligand_idx.reshape(-1, 1), axis=0)),
        torch.from_numpy(affinity.reshape(-1, 1)),
    )


def _load_ligands(path: pathlib.Path, encoder: Encoder) -> npt.NDArray[int]:
    with (path / "ligands.json").open("r") as ligands_file:
        ligands: dict[str, str] = json.load(ligands_file)
    return np.array([encoder.transform(x) for x in ligands.values()], dtype=int)


def _load_proteins(path: pathlib.Path, encoder: Encoder) -> npt.NDArray[int]:
    with (path / "proteins.json").open("r") as proteins_file:
        proteins: dict[str, str] = json.load(proteins_file)
    return np.array([encoder.transform(x) for x in proteins.values()], dtype=int)


def _load_affinity(path: pathlib.Path, normalize: bool = False) -> AffinityMatrix:
    affinity_scores = np.load(str(path / "affinity.npy"))
    if normalize:
        affinity_scores = _normalize(affinity_scores, base=10.0)
    return affinity_scores.astype(dtype=np.dtype("float32"))  # type: ignore


class AffinityDataset(torch.utils.data.TensorDataset):
    """Affinity dataset between ligands and proteins."""

    def __init__(
        self,
        path: pathlib.Path | None = None,
        normalize: bool = True,
        ligand_encoder: Encoder = LabelEncoder(table=_iso_smiles_table, n_dim=64),
        protein_encoder: Encoder = LabelEncoder(table=_protein_table, n_dim=512),
    ):
        self._ligand_encoder = ligand_encoder
        self._protein_encoder = protein_encoder

        if path is None:
            raise ValueError("Expecting a non-None path to load interaction from")
        proteins, ligands, affinity = _extract_interactions(
            proteins=_load_proteins(path, self._protein_encoder),
            ligands=_load_ligands(path, self._ligand_encoder),
            affinity=_load_affinity(path, normalize),
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
    dataset = AffinityDataset(path=pathlib.Path(DATA_PATH))

    protein = "MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKKTQPKARLTDPIPTTETSIAPRQRPKAGQTQPNPGILPIQPALTPRKRATVQPPPQAAGSSNQPGLLASVPQPKPQAPPSQPLPQTQAKQPQAPPTPQQTPSTQAQGLPAQAQATPQHQQQLFLKQQQQQQQPPPAQQQPAGTFYQQQQAQTQQFQAVHPATQKPAIAQFPVVSQGGSQQQLMQNFYQQQQQQQQQQQQQQLATALHQQQLMTQQAALQQKPTMAAGQQPQPQPAAAPQPAPAQEPAIQAPVRQQPKVQTTPPPAVQGQKVGSLTPPSSPKTQRAGHRRILSDVTHSAVFGVPASKSTQLLQAAAAEASLNKSKSATTTPSGSPRTSQQNVYNPSEGSTWNPFDDDNFSKLTAEELLNKDFAKLGEGKHPEKLGGSAESLIPGFQSTQGDAFATTSFSAGTAEKRKGGQTVDSGLPLLSVSDPFIPLQVPDAPEKLIEGLKSPDTSLLLPDLLPMTDPFGSTSDAVIEKADVAVESLIPGLEPPVPQRLPSQTESVTSNRTDSLTGEDSLLDCSLLSNPTTDLLEEFAPTAISAPVHKAAEDSNLISGFDVPEGSDKVAEDEFDPIPVLITKNPQGGHSRNSSGSSESSLPNLARSLLLVDQLIDL"
    print("Protein encoding:")
    print(f"{protein=} -> \n{dataset.encode_protein(protein)}")

    ligand = "CC1=C2C=C(C=CC2=NN1)C3=CC(=CN=C3)OCC(CC4=CC=CC=C4)N"
    print("Ligand encoding:")
    print(f"{ligand=} -> \n{dataset.encode_ligand(ligand)}")

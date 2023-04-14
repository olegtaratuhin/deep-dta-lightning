from __future__ import annotations

import pathlib
from typing import Callable, Protocol

import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from src.data.affinity_datasets import AffinityDataset, TabularAffinityDataset

__all__ = [
    "AffinityDataModule",
    "TabularAffinityDataModule",
    "PredictionDataModule",
]


class DatasetFactory(Protocol):
    def __call__(self, path: pathlib.Path) -> Dataset:
        raise NotImplementedError()


class AffinityDataModule(LightningDataModule):
    """Data module for binding affinity datasets.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        path: str = "data/",
        dataset: DatasetFactory = AffinityDataset,
        train_val_test_split: tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # this will init instance variables, so that linters won't complain
        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

    def prepare_data(self) -> None:
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = self.hparams.dataset(path=pathlib.Path(self.hparams.path))
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


class TabularAffinityDataModule(LightningDataModule):
    """Data module for tabular affinity datasets."""

    def __init__(
        self,
        path: str,
        loader: Callable[[str], pd.DataFrame] = pd.read_parquet,
        dataset: DatasetFactory = TabularAffinityDataset,
        train_val_test_split: tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # this will init instance variables, so that linters won't complain
        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

    def prepare_data(self) -> None:
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            df = self.hparams.loader(self.hparams.path)
            dataset = self.hparams.dataset(df)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


class PredictionDataModule(LightningDataModule):
    """Data module for prediction."""

    def __init__(
        self,
        path: str,
        loader: Callable[[str], pd.DataFrame] = pd.read_parquet,
        dataset: DatasetFactory = TabularAffinityDataset,
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # this will init instance variables, so that linters won't complain
        self.data_predict: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Load data.

        Set variable: `self.data_predict`.
        """
        # load dataset only if not loaded already
        if self.data_predict is None:
            df = self.hparams.loader(self.hparams.path)
            dataset = self.hparams.dataset(df)
            self.data_predict = dataset

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_predict,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

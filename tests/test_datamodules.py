import pytest
import torch

from src.data.affinity_datamodule import AffinityDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_davis_affinity_datamodule(batch_size):
    data_dir = "data/davis"

    dm = AffinityDataModule(path=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    x_protein, x_ligand, y_affinity = batch
    assert len(y_affinity) == batch_size
    assert len(y_affinity) == batch_size
    assert y_affinity.dtype == torch.float32
    assert x_protein.dtype == torch.int64
    assert x_ligand.dtype == torch.int64

import torch

from .datasets.datasets import (
    CUB200Dataset,
    CarsDataset,
    DogsDataset,
    FlowersDataset,
    NabirdsDataset,
    Food101Dataset,
)

_DATASET_CATALOG = {
    "CUB": CUB200Dataset,
    "OxfordFlowers": FlowersDataset,
    "StanfordCars": CarsDataset,
    "StanfordDogs": DogsDataset,
    "nabirds": NabirdsDataset,
    "food-101": Food101Dataset,
}


def _construct_loader(cfg, split, batch_size, shuffle, drop_last, transform):
    """Constructs the data loader for the given dataset."""
    dataset_name = cfg.DATA.NAME

    assert dataset_name in _DATASET_CATALOG.keys(), "Dataset '{}' not supported".format(
        dataset_name
    )
    print("Using dataset {}".format(dataset_name))
    dataset = _DATASET_CATALOG[dataset_name](cfg, split, transform=transform)

    # Create a sampler for multi-process training
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(True if shuffle and split == "train" else False),
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=cfg.DATA.PIN_MEMORY,
        drop_last=drop_last,
    )
    return loader

def build_train_loader(cfg, transform=None):
    return _construct_loader(
        cfg,
        split="train",
        batch_size=cfg.DATA.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        transform=transform,
    )

def build_test_loader(cfg, transform=None):
    return _construct_loader(
        cfg,
        split="test",
        batch_size=1,
        shuffle=False,
        drop_last=False,
        transform=transform,
    )
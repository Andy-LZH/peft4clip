import torch

from .datasets.datasets import (
    CUB200Dataset,
    CarsDataset,
    DogsDataset,
    FlowersDataset,
    NabirdsDataset,
    Food101Dataset,
)

from torchvision.datasets import SUN397

_DATASET_CATALOG = {
    "CUB": CUB200Dataset,
    "OxfordFlowers": FlowersDataset,
    "StanfordCars": CarsDataset,
    "StanfordDogs": DogsDataset,
    "nabirds": NabirdsDataset,
    "food-101": Food101Dataset,
    "vtab-sun397": SUN397,
}

import numpy as np


def _construct_loader(
    cfg, split, batch_size, shuffle, drop_last, transform, shots=-1, seed=0
):
    """Constructs the data loader for the given dataset."""
    dataset_name = cfg.DATA.NAME
    # Construct the dataset
    if dataset_name.startswith("vtab-"):
        # import the tensorflow here only if needed
        from .datasets.vtab import TFDataset

        if dataset_name not in ["vtab-sun397"]:
            dataset = TFDataset(cfg, split, transform=transform)
        else:
            dataset = _DATASET_CATALOG[dataset_name](root=cfg.DATA.DATAPATH, split=split, transform=transform)

    else:
        assert (
            dataset_name in _DATASET_CATALOG.keys()
        ), "Dataset '{}' not supported".format(dataset_name)
        dataset = _DATASET_CATALOG[dataset_name](cfg, split, transform=transform)

    if shots > 0 and split == "train":
        dataset = _few_shot_sampler(dataset, shots, seed)

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
        batch_size=cfg.DATA.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        transform=transform,
    )


def _few_shot_sampler(dataset, shots, seed):
    """Sampler for few-shot dataset."""
    # category data in each class (list of indices)
    category_data = [[] for _ in range(dataset.get_class_num())]
    for i, label in enumerate(dataset._targets):
        category_data[label].append(i)
    # check length of each class
    # randomly sample shots from each class
    np.random.seed(seed)
    sample_indices = []
    for indices in category_data:
        # if there are not enough data in this class, sample with replacement
        if len(indices) < shots:
            sample_indices.extend(np.random.choice(indices, shots, replace=True))
        else:
            sample_indices.extend(np.random.choice(indices, shots, replace=False))
    assert len(sample_indices) == shots * dataset.get_class_num()
    return torch.utils.data.Subset(dataset, sample_indices)


def construct_trainval_loader(cfg, transform=None, shots=0, seed=0):
    """Train loader wrapper."""
    if cfg.NUM_GPUS > 1:
        drop_last = True
    else:
        drop_last = False
    return _construct_loader(
        cfg=cfg,
        split="train",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=drop_last,
        transform=transform,
        shots=shots,
        seed=seed,
    )

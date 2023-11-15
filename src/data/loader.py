import torch
from tqdm import tqdm

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
        if dataset_name not in ["vtab-sun397"]:
            from .datasets.vtab import TFDataset
            dataset = TFDataset(cfg, split, transform=transform)

            if shots > 0 and split == "train":
                dataset = _few_shot_sampler(dataset, shots, seed)
        else:

            sun397_dataset = _DATASET_CATALOG[dataset_name](root=cfg.DATA.DATAPATH, transform=transform)

            # split the dataset into train and test randomly with 80% and 20% respectively use np.random.seed(0)
            np.random.seed(seed)
            indices = np.random.permutation(len(sun397_dataset))
            train_indices = indices[:int(0.8 * len(indices))]
            test_indices = indices[int(0.8 * len(indices)):]
            labels = np.array(sun397_dataset._labels)
            train_dataset = labels[train_indices]
            test_dataset = torch.utils.data.Subset(sun397_dataset, test_indices)
            if split == "train":
                dataset = train_dataset
                if shots > 0:
                    dataset = _few_shot_sampler(sun397_dataset, shots, seed, classes=cfg.DATA.CLASSES, targets=train_dataset)
            elif split == "test":
                dataset = test_dataset
    else:
        assert (
            dataset_name in _DATASET_CATALOG.keys()
        ), "Dataset '{}' not supported".format(dataset_name)
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
        batch_size=cfg.DATA.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        transform=transform,
    )


def _few_shot_sampler(dataset, shots, seed, classes=None, targets=None):
    """Sampler for few-shot dataset."""
    # category data in each class (list of indices)
    if classes is not None and targets is not None:
        category_data = [[] for _ in range(len(classes))]
        for i, label in tqdm(enumerate(targets), total=len(dataset), desc="few-shot sampler"):
            category_data[label].append(i)
        class_length = len(classes)
    else:
        category_data = [[] for _ in range(dataset.get_class_num())]
        for i, label in tqdm(enumerate(dataset._targets), total=len(dataset), desc="few-shot sampler"):
            category_data[label].append(i)
        class_length = dataset.get_class_num()
    
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
    assert len(sample_indices) == shots * class_length
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

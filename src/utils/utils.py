"""
Utility functions for the Auto-Adapter project

Edited upon code from https://github.com/kmnp/vpt
"""

import clip
import argparse
from src.configs.vit_configs import (
    get_b32_config,
    get_h14_config,
    get_b16_config,
)
from src.configs.config import get_cfg
from src.data.loader import (
    build_train_loader,
    build_test_loader,
    construct_trainval_loader,
)
import os

_DATASET_CONFIG = {
    "food-101": "configs/food-101.yaml",
    "CUB": "configs/cub.yaml",
    "OxfordFlowers": "configs/flowers.yaml",
    "StanfordCars": "configs/cars.yaml",
    "StanfordDogs": "configs/dogs.yaml",
    "vtab-caltech101": "configs/caltech101.yaml",
    "vtab-cifar100": "configs/cifar100.yaml",
    "vtab-dtd": "configs/dtd.yaml",
    "vtab-eurosat": "configs/eurosat.yaml",
    "vtab-oxford_pet": "configs/oxford_pet.yaml",
    "vtab-pcam": "configs/pcam.yaml",
    "vtab-svhncropped": "configs/svhncropped.yaml",
}


def setup_clip(args: argparse.Namespace) -> tuple:
    """
    Set up CLIP

    Parameters
    ----------
    args : argparse.Namespace
        Arguments from command line

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input

    model_config : dict
        The CLIP model config

    prompt_config : dict
        The CLIP prompt config
    """

    # check if model is valid
    if args.model not in ["ViT-B32", "ViT-B16", "ViT-L14"]:
        raise ValueError(
            "Model not supported yet, please choose from ViT-B32, ViT-B16, ViT-L14"
        )

    # check if device is valid
    if args.device not in ["cuda", "cpu"]:
        raise ValueError("Device not supported yet, please choose from cuda, cpu")

    # check if data is valid
    if args.data not in _DATASET_CONFIG.keys():
        print(args.data)
        raise ValueError(
            ("Dataset not supported yet, please choose from ", _DATASET_CONFIG.keys())
        )

    # set up model config
    if args.model == "ViT-B32":
        model_config = get_b32_config()
        model_type = "ViT-B/32"
    elif args.model == "ViT-B16":
        model_config = get_b16_config()
        model_type = "ViT-B/16"
    elif args.model == "ViT-L14":
        model_config = get_h14_config()
        model_type = "ViT-L/14"
    else:
        raise ValueError(
            "Model not supported yet, please choose from ViT-B32, ViT-B16, ViT-L14"
        )

    # set up CLIP
    model, preprocess = clip.load(model_type, device=args.device)

    # set up prompt config
    prompt_config = get_cfg().MODEL.PROMPT
    prompt_config.PROJECT = 768
    prompt_config.DEEP = args.deep

    # set up dataset config from yaml file

    cfg = get_cfg()
    cfg.merge_from_file(_DATASET_CONFIG[args.data])

    # read classes info from url from yaml file
    classes_path = cfg.DATA.CLASSESPATH
    print(classes_path)
    cfg.MODEL.TYPE = args.model
    if not os.path.exists(classes_path):
        raise ValueError(
            "Classes path not found, please check the path in the yaml file"
        )
    with open(classes_path, "r") as f:
        classes = f.readlines()
    classes = [c.strip() for c in classes]
    cfg.DATA.CLASSES = classes
    cfg.freeze()

    # set up dataset read from yaml file
    if args.data.startswith("vtab-"):
        train_loader = construct_trainval_loader(cfg, transform=preprocess)
    else:
        train_loader = build_train_loader(cfg, transform=preprocess)
    test_loader = build_test_loader(cfg, transform=preprocess)

    return model, model_config, prompt_config, train_loader, test_loader, cfg

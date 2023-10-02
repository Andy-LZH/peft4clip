"""
Utility functions for the Auto-Adapter project

Edited upon code from https://github.com/kmnp/vpt
"""

import os
import clip
import torch
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

from src.model.CLIP_VPT.VisionPromptCLIP import VisionPromptCLIP
from src.model.CLIP.VanillaCLIP import VanillaCLIP

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


def setup_model(args: argparse.Namespace) -> tuple:
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
    if args.model not in ["VPT-CLIP-Shallow", "VPT-CLIP-Deep", "VPT-CLIP-Linear"]:
        raise ValueError(
            "Model not supported yet, please choose from VPT-CLIP-Shallow, VPT-CLIP-Deep, VPT-CLIP-Linear"
        )

    if args.backbone not in ["ViT-B32", "ViT-B16", "ViT-L14"]:
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
    if args.backbone == "ViT-B32":
        model_config = get_b32_config()
        backbone_type = "ViT-B/32"
    elif args.backbone == "ViT-B16":
        model_config = get_b16_config()
        backbone_type = "ViT-B/16"
    elif args.backbone == "ViT-L14":
        model_config = get_h14_config()
        backbone_type = "ViT-L/14"
    else:
        raise ValueError(
            "Model not supported yet, please choose from ViT-B32, ViT-B16, ViT-L14"
        )

    # set up CLIP
    model, preprocess = clip.load(backbone_type, device=args.device)

    # set up prompt config
    prompt_config = get_cfg().MODEL.PROMPT
    prompt_config.PROJECT = 768

    # set up dataset config from yaml file

    cfg = get_cfg()
    file_location = _DATASET_CONFIG[args.data]

    # construct file locaation to get different yaml file for different model
    file_location = file_location[:8] + args.model + '/' + file_location[8:]
    print(file_location)
    cfg.merge_from_file(file_location)

    # read classes info from url from yaml file
    classes_path = cfg.DATA.CLASSESPATH
    print(classes_path)
    cfg.MODEL.TYPE = args.model
    cfg.MODEL.BACKBONE = args.backbone
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

    # return model, model_config, prompt_config, train_loader, test_loader, cfg
    return (
        _construct_model(
            args=args,
            model=model,
            model_config=model_config,
            prompt_config=prompt_config,
            dataset_config=cfg,
        ),
        train_loader,
        test_loader,
        cfg,
    )


def _construct_model(args, model, model_config, prompt_config, dataset_config):
    """
    Construct the model

    Parameters
    ----------
    args : argparse.Namespace
        Arguments from command line

    model : torch.nn.Module
        The CLIP model

    model_config : dict
        The CLIP model config

    prompt_config : dict
        The CLIP prompt config

    dataset_config : dict
        The dataset config
    """
    
    text_input = torch.cat(
        [clip.tokenize(f"a photo of {c}") for c in dataset_config.DATA.CLASSES]
    ).to(args.device)

    img_size = dataset_config.DATA.CROPSIZE
    num_classes = dataset_config.DATA.NUMBER_CLASSES

    if args.model == "VPT-CLIP-Shallow":
        return VisionPromptCLIP(
            backbone=model,
            config=model_config,
            dataset_config=dataset_config,
            prompt_config=prompt_config,
            img_size=img_size,
            num_classes=num_classes,
            prompts=text_input,
            deep=False,
        ).to(args.device)

    elif args.model == "VPT-CLIP-Deep":
        return VisionPromptCLIP(
            backbone=model,
            config=model_config,
            dataset_config=dataset_config,
            prompt_config=prompt_config,
            img_size=img_size,
            num_classes=num_classes,
            prompts=text_input,
            deep=True,
        ).to(args.device)

    elif args.model == "VPT-CLIP-Linear":
        return VanillaCLIP(
            backbone=model,
            config=model_config,
            prompt_config=prompt_config,
            img_size=img_size,
            num_classes=num_classes,
            prompts=text_input,
        ).to(args.device)

    else:
        raise ValueError(
            "Model not supported yet, please choose from VPT-CLIP-Shallow, VPT-CLIP-Deep, VPT-CLIP-Linear"
        )

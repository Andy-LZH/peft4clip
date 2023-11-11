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
    get_l16_config,
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
from src.model.CLIP_Adapter.Adapter import CLIP_Adapter
import open_clip

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
    "vtab-sun397": "configs/sun397.yaml",
    "vtab-clevr_count": "configs/clevr_count.yaml",
    "vtab-clevr_distance": "configs/clevr_distance.yaml",
    "vtab-dmlab": "configs/dmlab.yaml",
    "vtab-kitti": "configs/kitti.yaml",
    "vtab-smallnorb_azimuth": "configs/smallnorb_azimuth.yaml",
    "vtab-smallnorb_elevation": "configs/smallnorb_elevation.yaml",
    "vtab-dSprites_location": "configs/dSprites_location.yaml",
    "vtab-dSprites_orientation": "configs/dSprites_orientation.yaml",
}

_BACKBONE_CONFIG = {
    "MetaCLIP-B32-400M": "metaclip_400m",
    "MetaCLIP-B16-400M": "metaclip_400m",
    "MetaCLIP-L14-400M": "metaclip_400m",
    "MetaCLIP-B32-2.5B": "metaclip_fullcc",
    "MetaCLIP-B16-2.5B": "metaclip_fullcc",
    "MetaCLIP-L14-2.5B": "metaclip_fullcc",
    "MetaCLIP-H14-2.5B": "metaclip_fullcc",
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
    if args.backbone not in [
        "ViT-B32",
        "ViT-B16",
        "ViT-L14",
        "MetaCLIP-B32-400M",
        "MetaCLIP-B16-400M",
        "MetaCLIP-L14-400M",
        "MetaCLIP-B32-2.5B",
        "MetaCLIP-B16-2.5B",
        "MetaCLIP-L14-2.5B",
    ]:
        raise ValueError(
            "Model not supported yet, please choose from ViT-B32, ViT-B16, ViT-L14, MetaCLIP-B32-400M, MetaCLIP-B16-400M, MetaCLIP-L14-400M, MetaCLIP-B32-2.5B, MetaCLIP-B16-2.5B, MetaCLIP-L14-2.5B, MetaCLIP-H14-2.5B"
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

    if args.type not in ["vision", "vision-language"]:
        raise ValueError(
            "Prompt Type not supported yet, please choose from vision, vision-language"
        )

    # set up model config
    if args.backbone in ["ViT-B32", "MetaCLIP-B32-400M", "MetaCLIP-B32-2.5B"]:
        model_config = get_b32_config()
        open_clip_default = "ViT-B-32-quickgelu"
        clip_default = "ViT-B/32"
    elif args.backbone in ["ViT-B16", "MetaCLIP-B16-400M", "MetaCLIP-B16-2.5B"]:
        model_config = get_b16_config()
        open_clip_default = "ViT-B-16-quickgelu"
        clip_default = "ViT-B/16"
    elif args.backbone in ["ViT-L14", "MetaCLIP-L14-400M", "MetaCLIP-L14-2.5B"]:
        model_config = get_l16_config()
        open_clip_default = "ViT-L-14-quickgelu"
        clip_default = "ViT-L/14"
    else:
        raise ValueError(
            "Model not supported yet, please choose from ViT-B32, ViT-B16, ViT-L14"
        )

    # set up CLIP
    if args.backbone.startswith("MetaCLIP"):
        model, _, preprocess = open_clip.create_model_and_transforms(
            open_clip_default,
            pretrained=_BACKBONE_CONFIG[args.backbone],
            device=args.device,
        )
    else:
        model, preprocess = clip.load(clip_default, device=args.device)

    # set up prompt config
    prompt_config = get_cfg().MODEL.PROMPT
    prompt_config.PROJECT = 768

    # set up dataset config from yaml file

    cfg = get_cfg()
    file_location = _DATASET_CONFIG[args.data]

    # construct file locaation to get different yaml file for different model
    file_location = file_location[:8] + args.model + "/" + file_location[8:]
    print(file_location)
    cfg.merge_from_file(file_location)

    # read classes info from url from yaml file
    classes_path = cfg.DATA.CLASSESPATH
    print(classes_path)
    cfg.MODEL.TYPE = args.model + "-" + args.backbone + "-" + args.type
    cfg.MODEL.TRANSFER_TYPE = args.type
    cfg.MODEL.BACKBONE = args.backbone
    if not os.path.exists(classes_path):
        raise ValueError(
            "Classes path not found, please check the path in the yaml file"
        )
    with open(classes_path, "r") as f:
        classes = f.readlines()
    classes = [c.strip() for c in classes]
    cfg.DATA.CLASSES = classes
    cfg.DATA.SHOTS = args.shots
    cfg.freeze()

    # set up dataset read from yaml file
    if args.data.startswith("vtab-"): 
        train_loader = construct_trainval_loader(
            cfg, transform=preprocess, shots=args.shots, seed=args.seed
        )
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

    # setup text input template
    _DATASET_TEMPLATE = {
        "vtab-caltech101": "a photo of a {}",
        "vtab-cifar100": "a photo of a {}",
        "vtab-dtd": "a photo of a {}",
        "vtab-eurosat": "a photo of a {}",
        "vtab-oxford_pet": "a photo of a {}",
        "vtab-pcam": "a photo of a {}",
        "vtab-svhncropped": "a photo of a digital number {}",
        "vtab-sun397": "a photo of a {}",
        "vtab-clevr_count": "a photo of a {}",
        "vtab-clevr_distance": "a photo of a {}",
        "vtab-dmlab": "a photo of a {}",
        "vtab-kitti": "a photo with {}",
        "vtab-smallnorb_azimuth": "a photo of a {}",
        "vtab-smallnorb_elevation": "a photo of a {}",
        "vtab-dSprites_location": "a photo of a {}",
        "vtab-dSprites_orientation": "a photo of a {}",
    }

    if args.data in _DATASET_TEMPLATE.keys():
        text_input = torch.cat(
            [
                clip.tokenize(_DATASET_TEMPLATE[args.data].format(c))
                for c in dataset_config.DATA.CLASSES
            ]
        ).to(args.device)

    img_size = dataset_config.DATA.CROPSIZE
    num_classes = dataset_config.DATA.NUMBER_CLASSES

    _model_dict = {
        "VPT-CLIP-Shallow": VisionPromptCLIP(
            backbone=model,
            config=model_config,
            dataset_config=dataset_config,
            prompt_config=prompt_config,
            img_size=img_size,
            num_classes=num_classes,
            prompts=text_input,
            deep=False,
        ).to(args.device),
        "VPT-CLIP-Deep": VisionPromptCLIP(
            backbone=model,
            config=model_config,
            dataset_config=dataset_config,
            prompt_config=prompt_config,
            img_size=img_size,
            num_classes=num_classes,
            prompts=text_input,
            deep=True,
        ).to(args.device),
        "VPT-CLIP-Linear": VanillaCLIP(
            backbone=model,
            config=model_config,
            prompt_config=prompt_config,
            img_size=img_size,
            num_classes=num_classes,
            prompts=text_input,
        ).to(args.device),
        "CLIP-Adapter": CLIP_Adapter(
            backbone=model,
            config=model_config,
            prompt_config=prompt_config,
            prompts=text_input,
        ).to(args.device),
    }

    if args.model in _model_dict.keys():
        return _model_dict[args.model]

    else:
        raise ValueError(
            "Model not supported yet, please choose from VPT-CLIP-Shallow, VPT-CLIP-Deep, VPT-CLIP-Linear, CLIP-Adapter"
        )

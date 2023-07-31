# Utility functions for the Auto-Adapter project

import clip
import argparse
from src.model.vpt.src.configs.vit_configs import get_b32_config, get_h14_config, get_b16_config
from src.model.vpt.src.configs.config import get_cfg

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
    if args.model not in ["ViT-B/32", "ViT-B/16", "ViT-L/14"]:
        raise ValueError("Model not supported yet, please choose from ViT-B/32, ViT-B/16, ViT-L/14")
    
    # check if device is valid
    if args.device not in ["cuda", "cpu"]:
        raise ValueError("Device not supported yet, please choose from cuda, cpu")
    
    # check if data is valid
    if args.data not in ["Rice_Image_Dataset"]:
        raise ValueError("Dataset not supported yet, please choose from Rice_Image_Dataset")
    
    # set up CLIP
    model, preprocess = clip.load(args.model, device=args.device)

    # set up model config
    if args.model == "ViT-B/32":
        model_config = get_b32_config()
    elif args.model == "ViT-B/16":
        model_config = get_b16_config()
    elif args.model == "ViT-L/14":
        model_config = get_h14_config()
    else:
        raise ValueError("Model not supported yet, please choose from ViT-B/32, ViT-B/16, ViT-L/14")
    
    # set up prompt config
    prompt_config = get_cfg().MODEL.PROMPT
    prompt_config.PROJECT = 768

    return model, preprocess, model_config, prompt_config
    
    

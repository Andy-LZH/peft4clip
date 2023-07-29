# Utility functions for the Auto-Adapter project

import os
import sys
import argparse

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
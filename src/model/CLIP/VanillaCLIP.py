"""
Edited upon code from https://github.com/kmnp/vpt
"""
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair


class VanillaCLIP(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        config,
        prompt_config,
        prompts,
        img_size=224,
        num_classes=5,
    ):
        super().__init__()  # python3 syntax

        print("Setting up CLIP Linear...")
        print("Setting up prompt configs...")
        assert prompt_config.LOCATION == "prepend"
        assert prompt_config.INITIATION == "random"
        assert prompt_config.NUM_DEEP_LAYERS is None
        assert not prompt_config.DEEP_SHARED

        print("Setting up CLIP model...")
        # get configs
        self.vit_config = config
        self.prompt_config = prompt_config
        self.prompts = prompts

        # set vit configs
        self.model = backbone  # temporary fix, need to be more general
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # convert to tuple if not, e.g. 224 -> (224, 224)
        self.img_size = _pair(img_size)
        # tuple of patch size, e.g. (16, 16)
        self.patch_size = self.vit_config.patches.size
        self.ViT = self.model.visual

        # set prompt configs
        self.num_tokens = self.prompt_config.NUM_TOKENS
        self.prompt_dropout = nn.Dropout(0.2)

        # output layer
        self.head = nn.Linear(self.ViT.output_dim, num_classes)

        # print("Setting up prompt...")
        # print("Project: ", self.prompt_config.PROJECT)

    def build_optimizer(self, configs):
        optimizer = torch.optim.AdamW(
            self.prompt_parameters,
            lr=configs.SOLVER.BASE_LR,
            weight_decay=configs.SOLVER.WEIGHT_DECAY,
        )
        return optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Vision Prompt CLIP

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor

        Returns
        -------
        logits : torch.Tensor
            Output logits tensor
        """
        # retrive image features
        img_features = self.ViT(x)
        return self.head(img_features)

"""
Edited upon code from https://github.com/kmnp/vpt
"""
import clip
import torch
import math
import torch.nn as nn
from time import sleep
from operator import mul
from functools import reduce
from torch.nn.modules.utils import _pair
from src.model.vpt.src.models.vit_backbones.vit import Embeddings


class VisionPromptCLIP(nn.Module):
    def __init__(self, backbone: nn.Module, config, prompt_config, img_size=224, num_classes=5):
        super().__init__()  # python3 syntax

        print("Setting up prompt configs...")
        assert prompt_config.DEEP == False, "Deep prompt not supported yet"
        assert prompt_config.LOCATION == "prepend"
        assert prompt_config.INITIATION == "random"
        assert prompt_config.NUM_DEEP_LAYERS is None
        assert not prompt_config.DEEP_SHARED

        print("Setting up CLIP model...")
        # get configs
        self.vit_config = config
        self.prompt_config = prompt_config

        # set vit configs
        self.model = backbone  # temporary fix, need to be more general
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # convert to tuple if not, e.g. 224 -> (224, 224)
        self.img_size = _pair(img_size)
        # tuple of patch size, e.g. (16, 16)
        self.patch_size = self.vit_config.patches.size
        self.ViT = self.model.visual
        self.embeddings = Embeddings(self.vit_config, self.img_size)

        # set prompt configs
        self.num_tokens = self.prompt_config.NUM_TOKENS
        self.prompt_dropout = nn.Dropout(self.prompt_config.DROPOUT)

        # if project the prompt embeddings
        if self.prompt_config.PROJECT > -1:
            # only for prepend / add
            self.prompt_dim = self.prompt_config.PROJECT
            self.prompt_proj = nn.Linear(
                self.prompt_dim, config.hidden_size)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            self.prompt_dim = config.hidden_size
            self.prompt_proj = nn.Identity()

        # initiate prompt:
        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_size, 1) + self.prompt_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, self.num_tokens, self.prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.prompt_config.DEEP:  # noqa
                total_d_layer = config.transformer["num_layers"]-1
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, self.num_tokens, self.prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")

    def incorporate_prompt(self, x):
        """
        Incorporate prompt into input based on https://github.com/kmnp/vpt

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor

        Returns
        -------
        embedding : torch.Tensor
            TODO change below description
            Input image tensor with shape: (batch_size, 1 + n_patches + n_prompt, hidden_dim)
        """
        B = x.shape[0]
        # after CLS token, all before image patches
        x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        embedding = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(self.prompt_proj(
                self.prompt_embeddings).expand(B, -1, -1)),
            x[:, 1:, :]
        ), dim=1)  # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        return embedding

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

        # convert to tensor if not
        img = x.to(self.device)

        # incorporate prompt
        incoporated_prompt = self.incorporate_prompt(img)

        # forward pass through transformer
        transformer = self.ViT.transformer

        # convert to fp16 to be comatible with CLIP
        x = transformer(incoporated_prompt.half())

        # Layernorm
        x = self.ViT.ln_post(x[:, 0, :])  # (batch_size, hidden_dim)

        # project to output dim
        if self.ViT.output_dim != x.shape[-1]:
            x = x @ self.ViT.proj  # (batch_size, output_dim)
        
        # return logits
        return x

    def train(self):
        """
        Set model to train mode, remain CLIP frozen
        """

        # freeze CLIP
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # set model to train mode
        self.prompt_proj.train()
        self.prompt_dropout.train()

    def eval(self):
        """
        Set model to eval mode, remain CLIP frozen
        """

        # freeze CLIP
        for param in self.parameters():
            param.requires_grad = False
        self.model.eval()

        # set model to eval mode
        for module in self.children():
            module.eval()
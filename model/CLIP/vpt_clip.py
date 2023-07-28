"""
Edited upon code from https://github.com/kmnp/vpt
"""
import clip
import torch
import math
import torch.nn as nn
from operator import mul
from functools import reduce
from torch.nn.modules.utils import _pair
from model.vpt.src.models.vit_backbones.vit import Embeddings


class VisionPromptCLIP(nn.Module):
    def __init__(self, backbone, config, prompt_config, img_size=224, num_classes=5):
        super().__init__() # python3 syntax

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
        self.model = backbone # temporary fix, need to be more general
        self.img_size = _pair(img_size) # convert to tuple if not, e.g. 224 -> (224, 224)
        self.patch_size = self.vit_config.patches.size # tuple of patch size, e.g. (16, 16)
        self.encoder = self.model.visual
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
        """
        B = x.shape[0]
        # after CLS token, all before image patches
        x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                x[:, 1:, :]
            ), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        return x

    def forward(self, x):
        # this is the default version:
        embedding_output = self.incorporate_prompt(x)
        clip_output = self.model.encode_image(x.to("cuda"))

        print("embedding_output.shape: ", embedding_output.shape)
        print("clip_output.shape: ", clip_output.shape)
        # contrastive loss on both image and text
        return embedding_output, clip_output
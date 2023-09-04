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
from src.model.vpt_clip.backbones.clip_embedding import CLIPEmbedding


class VisionPromptCLIP(nn.Module):
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
        self.embeddings = CLIPEmbedding(self.ViT, self.vit_config, self.img_size)

        # set prompt configs
        self.num_tokens = self.prompt_config.NUM_TOKENS
        self.prompt_dropout = nn.Dropout(0.2)

        # output layer
        self.head = nn.Linear(self.ViT.output_dim, num_classes)

        print("Setting up prompt...")
        print("Project: ", self.prompt_config.PROJECT)

        sleep(1)
        # if project the prompt embeddings
        if self.prompt_config.PROJECT > -1:
            # only for prepend / add
            print("Project")
            self.prompt_dim = self.prompt_config.PROJECT
            self.prompt_proj = nn.Linear(self.prompt_dim, config.hidden_size)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode="fan_out")
        else:
            print("No project")
            self.prompt_dim = config.hidden_size
            self.prompt_proj = nn.Identity()

        # initiate prompt:
        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(
                6.0 / float(3 * reduce(mul, self.patch_size, 1) + self.prompt_dim)
            )  # noqa

            self.prompt_embeddings = nn.Parameter(
                torch.zeros(1, self.num_tokens, self.prompt_dim)
            )
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.prompt_config.DEEP:  # noqa
                total_d_layer = config.transformer["num_layers"] - 1
                self.deep_prompt_embeddings = nn.Parameter(
                    torch.zeros(total_d_layer, self.num_tokens, self.prompt_dim)
                )
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

        embedding = torch.cat(
            (
                x[:, :1, :],
                self.prompt_dropout(
                    self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)
                ),
                x[:, 1:, :],
            ),
            dim=1,
        )  # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        # check dim-1 n_prompt
        return embedding

    def encode_image(self, image):
        img = image
        # incorporate prompt
        incoporated_prompt = self.incorporate_prompt(
            img
        )  # (batch_size, 1 + n_patches + n_prompt, hidden_dim)

        # reform embedding to match CLIP
        incoporated_prompt = incoporated_prompt.permute(
            1, 0, 2
        )  # (1 + n_patches + n_prompt, batch_size, hidden_dim)

        # make sure self.ViT.transformer not update but incoporated_prompt can
        input_embedding = self.ViT.transformer(incoporated_prompt)

        # reform embedding to perform loss calculation
        input_embedding = input_embedding.permute(
            1, 0, 2
        )  # (batch_size, 1 + n_patches + n_prompt, hidden_dim)
        # Layernorm

        input_embedding = self.ViT.ln_post(
            input_embedding[:, 0, :]
        )  # (batch_size, hidden_dim)

        # project to output dim
        if self.ViT.output_dim != input_embedding.shape[-1]:
            input_embedding = (
                input_embedding @ self.ViT.proj
            )  # (batch_size, output_dim)

        image_features_vpt = input_embedding

        return image_features_vpt

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
        image_features_vpt = self.encode_image(x)

        with torch.no_grad():
            text_features = self.model.encode_text(self.prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale
        logits = logit_scale * (image_features_vpt @ text_features.t())
        self.prompt_proj.train()
        self.prompt_dropout.train()
        return logits

    def forward_deep_prompt(self, embedding_output):
        """
        Forward pass of Vision Prompt CLIP with deep prompt

        Parameters
        ----------
        `embedding_output` : torch.Tensor
            input embedding tensor

        Returns
        -------
        `encoded` : torch.Tensor
            encoded tensor
        `attn_weights` : list
            list of attention weights
        """
        # setup for deep prompt
        attn_weights = []  # query, key, value
        hidden_states = None
        weights = None
        B = embedding_output.shape[0]  # hidden_dim
        num_layers = self.ViT.transformer.layers

        for i in range(num_layers):
            if i == 0:
                hidden_states = self.ViT.transformer.resblocks[i](embedding_output)

            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(
                        self.prompt_proj(self.deep_prompt_embeddings[i - 1]).expand(
                            B, -1, -1
                        )
                    )

                    hidden_states = torch.cat(
                        (
                            hidden_states[:, :1, :],
                            deep_prompt_emb,
                            hidden_states[:, (1 + self.num_tokens) :, :],
                        ),
                        dim=1,
                    )

                hidden_states = self.ViT.transformer.resblocks[i](hidden_states)

        hidden_states = hidden_states.permute(1, 0, 2) # (batch_size, seq_len, dim)
        input_embedding = self.ViT.ln_post(
            hidden_states[:, 0, :]
        )  # (batch_size, hidden_dim)

        # project to output dim
        if self.ViT.output_dim != input_embedding.shape[-1]:
            encoded = (
                input_embedding @ self.ViT.proj
            )  # (batch_size, output_dim)
        return encoded

    def linear_probe(self, x: torch.Tensor) -> torch.Tensor:
        """
        Linear probe of Vision Prompt CLIP
        """
        if self.prompt_config.DEEP:
            img = x
            # incorporate prompt
            incoporated_prompt = self.incorporate_prompt(
                img
            )  # (batch_size, 1 + n_patches + n_prompt, hidden_dim)

            # reform embedding to match CLIP
            incoporated_prompt = incoporated_prompt.permute(
                1, 0, 2
            )  # (1 + n_patches + n_prompt, batch_size, hidden_dim)

            encoded = self.forward_deep_prompt(incoporated_prompt)
        else:
            encoded = self.encode_image(x)

        self.head.train()
        logits = self.head(encoded)
        return logits

    def train(self):
        """
        Set model to train mode, remain CLIP frozen
        """

        # freeze CLIP
        self.model.eval()
        self.ViT.eval()
        self.ViT.transformer.eval()
        self.embeddings.eval()

        # set model to train mode
        self.prompt_proj.train()
        self.prompt_dropout.train()
        self.head.train()

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

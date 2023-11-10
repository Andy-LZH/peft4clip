"""
Borrowed code from https://github.com/kmnp/vpt
"""
import math
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from functools import reduce
from operator import mul
from src.model.CLIP_VPT.Embeddings import CLIPInputEmbedding


class VisionPromptCLIP(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        config,
        dataset_config,
        prompt_config,
        prompts,
        deep=False,
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

        # set prompt configs
        self.num_tokens = dataset_config.MODEL.PROMPT.NUM_TOKENS
        self.prompt_dropout = nn.Dropout(0.1)
        self.deep = deep

        print("Setting up prompt...")
        print("Project: ", self.prompt_config.PROJECT)

        # if project the prompt embeddings
        if self.prompt_config.PROJECT > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config.PROJECT
            self.prompt_proj = nn.Linear(prompt_dim, config.hidden_size)
            nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode="fan_out")
        else:
            prompt_dim = config.hidden_size
            self.prompt_proj = nn.Identity()

        # initiate prompt:
        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(
                6.0 / float(3 * reduce(mul, self.patch_size, 1) + prompt_dim)
            )  # noqa

            self.prompt_embeddings = nn.Parameter(
                torch.zeros(1, self.num_tokens, prompt_dim)
            )

            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.deep:
                total_d_layer = self.ViT.transformer.layers - 1
                self.deep_prompt_embeddings = nn.Parameter(
                    torch.zeros(total_d_layer, self.num_tokens, prompt_dim)
                )
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

        # set embedding layer
        self.embeddings = CLIPInputEmbedding(ViT=self.ViT)

        # output layer
        self.head = nn.Linear(self.ViT.output_dim, num_classes)

    def build_optimizer(self, configs):
        params = (
            list(self.head.parameters())
            + list(self.prompt_dropout.parameters())
            + list(self.prompt_proj.parameters())
        )

        optimizer = torch.optim.SGD(
            params=params,
            lr=configs.SOLVER.BASE_LR,
            weight_decay=configs.SOLVER.WEIGHT_DECAY,
            momentum=configs.SOLVER.MOMENTUM,
        )

        return optimizer

    def incorporate_prompt(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.embeddings(x)  # (batch_size, num_patches**2 + 1, hidden_dim)

        self.prompt_embed_multi_dim = self.prompt_embeddings.expand(batch_size, -1, -1)

        # add learnable context prompt of shape (1, num_tokens, hidden_dim)
        # (batch_size, 1 + num_tokens + num_patches**2, hidden_dim))
        x = torch.cat(
            (
                x[:, :1, :],
                self.prompt_dropout(self.prompt_proj(self.prompt_embed_multi_dim)),
                x[:, 1:, :],
            ),
            dim=1,
        )

        # reshape to (1 + num_tokens + num_patches**2, batch_size, hidden_dim) for matmul
        x = x.permute(1, 0, 2)

        return x

    def forward_shallow(self, embedding: torch.Tensor) -> torch.Tensor:
        # feed to transformer
        x = self.ViT.transformer(embedding)

        # reformate to (batch_size, 1 + num_tokens + num_patches**2, hidden_dim)
        x = x.permute(1, 0, 2)

        # take the first token + extra token
        x = x[:, 0, :]

        x = self.ViT.ln_post(x)

        if self.ViT.proj is not None:
            x = x @ self.ViT.proj

        return self.head(x)

    def forward_deep(self, embedding_output: torch.Tensor) -> torch.Tensor:
        hidden_states = None
        num_layers = self.ViT.transformer.layers
        B = embedding_output.shape[1]

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
                    # reshape to (num_tokens, batch_size, hidden_dim) for matmul
                    deep_prompt_emb = deep_prompt_emb.permute(1, 0, 2)

                # since now in clip embedding_output is (1 + num_tokens + num_patches**2, batch_size, hidden_dim)
                # we need to change directly on the dimension 0
                hidden_states = torch.cat(
                    (
                        hidden_states[:1, :, :],
                        deep_prompt_emb,
                        hidden_states[(1 + self.num_tokens) :, :, :],
                    ),
                    dim=0,
                )

                hidden_states = self.ViT.transformer.resblocks[i](hidden_states)

        # reformate to (batch_size, 1 + num_tokens + num_patches**2, hidden_dim)
        hidden_states = hidden_states.permute(1, 0, 2)
        hidden_states = hidden_states[:, 0, :]
        hidden_states = self.ViT.ln_post(hidden_states)

        if self.ViT.proj is not None:
            hidden_states = hidden_states @ self.ViT.proj

        return self.head(hidden_states)

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
        # incorporate prompt
        x = self.incorporate_prompt(x)

        if self.deep:
            return self.forward_deep(x)
        else:
            return self.forward_shallow(x)

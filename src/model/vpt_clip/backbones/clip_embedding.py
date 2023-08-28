import torch
import torch.nn as nn
from src.model.vpt.src.models.vit_backbones.vit import Embeddings


class CLIPEmbedding(Embeddings):
    """
    CLIP Embedding module, inherit from Embeddings module in ViT backbone
    """

    def __init__(self, ViT, config, img_size, in_channels=3):
        """
        Constructor of CLIP Embedding module

        Parameters
        ----------
        ViT : nn.Module
            CLIP vision encoder
        config : dict
            Configurations of CLIP
        img_size : tuple
            Size of input image
        in_channels : int, optional
            Number of input channels, by default 3
        """
        super().__init__(config, img_size, in_channels=in_channels)

        # set CLIP configs
        self.original_patch_embeddings = self.patch_embeddings
        self.patch_embeddings = ViT.conv1
        self.input_resolution = img_size
        self.ln_pre = ViT.ln_pre

    def forward(self, x):
        """
        Construct to expacted input embedding for Transformer in CLIP
        @reference github.com/openai/CLIP
        """
        # retrive patch embeddings
        with torch.no_grad():
            x = self.patch_embeddings(x)  # shape = [batch, width, grid, grid]

        # from patch embeddings get width and form scale
        width = x.shape[1]
        scale = width**-0.5
        patch_size = x.shape[2]
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale
            * torch.randn((self.input_resolution[0] // patch_size) ** 2 + 1, width)
        )

        # reform embeddings
        # shape = [batch, width, grid*grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [batch, n_patches, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.device)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, hidden_path + 1, width]
        x = x + self.position_embeddings.to(x.dtype)
        x = self.ln_pre(x)
        return x

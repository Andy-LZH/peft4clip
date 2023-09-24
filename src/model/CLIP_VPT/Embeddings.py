import torch
from torch import nn


class CLIPInputEmbedding(nn.Module):
    """
    CLIP Input Embedding Module
    """

    def __init__(self, ViT) -> None:
        """
        Constructor method

        Parameters
        ----------

        """
        super().__init__()

        # define embedding layers
        self.ViT = ViT
        self.patch_embedding = (
            self.ViT.conv1
        )  # nn.Conv2d(3, self.hidden_dim, self.patch_size, self.patch_size)
        self.ln_pre = self.ViT.ln_pre
        self.positional_embedding = self.ViT.positional_embedding
        self.class_embedding = self.ViT.class_embedding

    def forward(self, x: torch.Tensor):
        """
        Returns pre_constructed CLIP input embedding to be added to the prompt embedding
        """

        # extract patches
        x = self.patch_embedding(
            x
        )  # (batch_size, hidden_dim, num_patches, num_patches)

        # debug print
        # print("Patch Embedding: ", x.shape)

        # flatten patches to let num_patches**2 be the sequence length
        # now the sequence length is num_patches**2 and the hidden dimension is hidden_dim
        x = x.reshape(
            x.shape[0], x.shape[1], -1
        )  # (batch_size, hidden_dim, num_patches**2)
        x = x.permute(0, 2, 1)  # (batch_size, num_patches**2, hidden_dim)

        # debug print
        # print("Patch Embedding: ", x.shape)

        # add cls token
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]

        # debug print
        # print("Patch Embedding: ", x.shape)

        # add positional embedding
        x = x + self.positional_embedding.to(x.dtype)

        # layer norm
        return self.ln_pre(x)
import torch
from torch import nn


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class CLIP_Adapter(nn.Module):
    def __init__(self, backbone, config, prompt_config, prompts):
        super(CLIP_Adapter, self).__init__()
        self.prompt_config = prompt_config
        self.model_config = config
        self.model = backbone
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adapter = Adapter(self.model.visual.output_dim)
        self.prompts = prompts
        self.alpha = 0.5

    def build_optimizer(self, args):
        optimizer = torch.optim.AdamW(
            self.adapter.parameters(),
            lr=args.SOLVER.BASE_LR,
            weight_decay=args.SOLVER.WEIGHT_DECAY,
        )
        return optimizer

    def forward(self, x):
        """
        Currently alpha set to 0.5 for half and half
        """
        img_embedding = self.model.encode_image(x)
        new_feature = self.adapter(img_embedding)
        new_feature = self.alpha * new_feature + (1 - self.alpha) * img_embedding
        return new_feature

    def vision_language_forward(self, x):
        """
        Currently alpha set to 0.5 for half and half
        """
        img_embedding = self.model.encode_image(x)
        new_feature = self.adapter(img_embedding)
        new_feature = self.alpha * new_feature + (1 - self.alpha) * img_embedding

        # calculate logits via contrastive loss between prompt and image
        image_features = new_feature / new_feature.norm(dim=-1, keepdim=True)
        text_features = self.model.encode_text(self.prompts) / self.prompts.norm(
            dim=-1, keepdim=True
        )
        logits_per_image = image_features @ text_features.t()
        return logits_per_image

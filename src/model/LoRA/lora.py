import torch
from torch import nn

class CLIP_LoRA(nn.Module):
    def __init__(self, backbone, config, prompt_config, prompts):
        super(CLIP_LoRA, self).__init__()
        self.prompt_config = prompt_config
        self.model_config = config
        self.model = backbone
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    def _replace_with_lora(self, model):
        for name, module in model.named_children():

import torch
from torch import nn
import torch.nn.functional as F


class SoftmaxLoss(nn.Module):
    def __init__(self, cfg=None):
        super(SoftmaxLoss, self).__init__()

    def loss(self, logits, targets, per_cls_weights, kwargs):
        weight = torch.tensor(per_cls_weights, device=logits.device)
        loss = F.cross_entropy(logits, targets, weight, reduction="none")

        return torch.sum(loss) / targets.shape[0]

    def forward(self, pred_logits, targets, per_cls_weights, multihot_targets=False):
        loss = self.loss(pred_logits, targets, per_cls_weights, multihot_targets)
        return loss
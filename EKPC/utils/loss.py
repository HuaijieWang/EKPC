import torch.nn as nn
from torch.nn import functional as F


class AngularPenaltySMLoss(nn.Module):
    def __init__(self, s=20.0, m=0.01):
        super().__init__()
        self.s = s
        self.m = m

    def forward(self, logits, labels):
        margin = (
            F.one_hot(labels, num_classes=logits.shape[1]).to(logits.dtype) * self.m
        )
        return F.cross_entropy(self.s * (logits - margin), labels)

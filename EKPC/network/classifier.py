import torch
from timm.models.layers.weight_init import trunc_normal_
from torch import nn
from torch.nn import functional as F


class SimpleContinualLinear(nn.Module):
    def __init__(self, embed_dim, nb_classes, feat_expand=False, with_norm=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.feat_expand = feat_expand
        self.with_norm = with_norm
        heads = []
        single_head = []
        if with_norm:
            single_head.append(nn.LayerNorm(embed_dim))
        single_head.append(nn.Linear(embed_dim, nb_classes, bias=False))
        head = nn.Sequential(*single_head)
        heads.append(head)
        self.heads = nn.ModuleList(heads)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)

    def update(self, nb_classes, freeze_old=True):
        single_head = []
        if self.with_norm:
            single_head.append(nn.LayerNorm(self.embed_dim))
        _fc = nn.Linear(self.embed_dim, nb_classes, bias=False)
        trunc_normal_(_fc.weight, std=0.02)
        single_head.append(_fc)
        new_head = nn.Sequential(*single_head)
        if freeze_old:
            for p in self.heads.parameters():
                p.requires_grad = False
        self.heads.append(new_head)

    def forward(self, x):
        out = []
        for ti in range(len(self.heads)):
            fc_inp = x[ti] if self.feat_expand else x
            out.append(
                1
                * F.linear(
                    F.normalize(fc_inp, p=2, dim=1),
                    F.normalize(self.heads[ti][0].weight, p=2, dim=1),
                )
            )
        out = {"logits": torch.cat(out, dim=1)}
        return out

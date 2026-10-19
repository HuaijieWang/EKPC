import torch
from torch.nn import functional as F


class HiddenImp:
    def __init__(self, net, device):
        self.net = net
        self.device = device
        self.mean_feature_mag = [
            torch.zeros(adapter.down_proj.out_features, device=device)
            for adapter in net
        ]
        self.util = [torch.zeros_like(value) for value in self.mean_feature_mag]
        self.begin_task()

    def begin_task(self):
        self._down_sum = [torch.zeros_like(value) for value in self.mean_feature_mag]
        self._up_sum = [torch.zeros_like(value) for value in self.util]
        self._count = 0

    @torch.no_grad()
    def obain_imp(self, features):
        if len(features) != len(self.net):
            raise ValueError("Expected one hidden activation tensor per adapter")
        batch_size = features[0].shape[0]
        for idx, (adapter, hidden) in enumerate(zip(self.net, features)):
            routing = F.cosine_similarity(hidden, hidden[:, :1, :], dim=-1)
            units = (hidden * routing.unsqueeze(-1)).sum(dim=1)
            self._down_sum[idx] += units.sum(dim=0)
            outgoing = adapter.up_proj.weight.detach().abs().sum(dim=0)
            self._up_sum[idx] += (units * outgoing).sum(dim=0)
        self._count += batch_size

    def finalize_task(self):
        if self._count == 0:
            raise ValueError("Cannot estimate local importance from an empty task")
        for idx in range(len(self.net)):
            self.mean_feature_mag[idx] += self._down_sum[idx] / self._count
            self.util[idx] += self._up_sum[idx] / self._count

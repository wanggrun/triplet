import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from .. import builder

@HEADS.register_module
class TripletLossHead(nn.Module):
    """Head for contrastive learning.
    """

    def __init__(self, predictor, gamma=2, size_average=True):
        super(TripletLossHead, self).__init__()
        self.predictor = builder.build_neck(predictor)
        self.size_average = size_average
        self.ranking_loss = nn.MarginRankingLoss(margin=100.)
        self.gamma = gamma

    def init_weights(self, init_linear='normal'):
        self.predictor.init_weights(init_linear=init_linear)

    def forward(self, input, target):
        pred = self.predictor([input])[0]
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        n = input.size(0)
        dist = -2. * torch.matmul(pred_norm, target_norm.t())
        idx = torch.arange(n)
        mask = idx.expand(n, n).eq(idx.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            down_k, _ = torch.topk(dist[i][mask[i]==0], 10, dim=-1, largest=False)
            down_k = down_k[1:].mean().unsqueeze(0)
            dist_an.append(down_k)
            # dist_an.append(dist[i][mask[i] == 0].median().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        loss_triplet = self.ranking_loss(dist_an, self.gamma * dist_ap, y)
        return dict(loss=loss_triplet)
            

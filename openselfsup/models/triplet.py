import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS

import numpy as np
import torch.nn.functional as F

@MODELS.register_module
class Triplet(nn.Module):
    """Triplet.

    Implementation of "Bootstrap Your Own Latent: A New Approach to
    Self-Supervised Learning (https://arxiv.org/abs/2006.07733)".

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        base_momentum (float): The base momentum coefficient for the target network.
            Default: 0.996.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 base_momentum=0.996,
                 **kwargs):
        super(Triplet, self).__init__()
        self.online_net = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.target_net = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.backbone = self.online_net[0]
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)

        self.base_momentum = base_momentum
        self.momentum = base_momentum

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.online_net[0].init_weights(pretrained=pretrained) # backbone
        self.online_net[1].init_weights(init_linear='kaiming') # projection
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data.copy_(param_ol.data)
        # init the predictor in the head
        self.head.init_weights()

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of the target network."""
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + \
                             param_ol.data * (1. - self.momentum)

    @torch.no_grad()
    def momentum_update(self):
        self._momentum_update()


    def _batch_shuffle_ddp(self, x):
        """Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        #x_gather= torch.cat(GatherLayer.apply(x), dim=0)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        #x_gather=torch.cat(GatherLayer.apply(x), dim=0)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
   
        img_v1 = img[:, 0, ...].contiguous()
        img_v2 = img[:, 1, ...].contiguous()
        # compute query features

        resize_size = 96 + np.random.randint(129, size=1)
        img_v1_online = F.interpolate(img_v1, size=[resize_size[0], resize_size[0]], mode="bilinear")
        proj_online_v1 = self.online_net(img_v1_online)[0]

        resize_size = 96 + np.random.randint(129, size=1)
        img_v2_online = F.interpolate(img_v2, size=[resize_size[0], resize_size[0]], mode="bilinear")
        proj_online_v2 = self.online_net(img_v2_online)[0]
        with torch.no_grad():
            #img_v1, idx_unshuffle = self._batch_shuffle_ddp(img_v1)
            resize_size = 128 + np.random.randint(97, size=1)
            img_v2_target = F.interpolate(img_v2, size=[resize_size[0], resize_size[0]], mode="bilinear")
            proj_target_v2 = self.target_net(img_v2_target)[0].clone().detach()
            #proj_target_v1 = self._batch_unshuffle_ddp(proj_target_v1, idx_unshuffle)
            
            #img_v2, idx_unshuffle = self._batch_shuffle_ddp(img_v2)
            resize_size = 128 + np.random.randint(97, size=1)
            img_v1_target = F.interpolate(img_v1, size=[resize_size[0], resize_size[0]], mode="bilinear")
            proj_target_v1 = self.target_net(img_v1_target)[0].clone().detach()
            #proj_target_v2 = self._batch_unshuffle_ddp(proj_target_v2, idx_unshuffle)

        loss = self.head(proj_online_v1, proj_target_v2)['loss'] + \
               self.head(proj_online_v2, proj_target_v1)['loss']
        return dict(loss=loss)

    def forward_test(self, img, **kwargs):
        pass
        print(key)

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))


@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

"""ResNet backbone with frozen BatchNorm, returning a stride-32 feature map and its padding mask."""

from __future__ import annotations

from collections.abc import Sequence

import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.ops.misc import FrozenBatchNorm2d


class Backbone(nn.Module):
    """ResNet trunk (``conv1`` ... ``layer4``) with ``FrozenBatchNorm2d``.

    Batch statistics from small, heavily padded detection batches are noisy, so BatchNorm
    keeps its ImageNet statistics. Only ``train_layers`` receive gradients.
    """

    def __init__(self, name: str = "resnet50", pretrained: bool = True,
                 train_layers: Sequence[str] = ("layer2", "layer3", "layer4")):
        super().__init__()
        weights = torchvision.models.get_model_weights(name).DEFAULT if pretrained else None
        resnet = getattr(torchvision.models, name)(weights=weights, norm_layer=FrozenBatchNorm2d)
        for n, p in resnet.named_parameters():
            p.requires_grad_(any(n.startswith(layer) for layer in train_layers))
        self.body = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                  resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.num_channels = resnet.fc.in_features

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        feat = self.body(x)
        feat_mask = F.interpolate(mask[None].float(), size=feat.shape[-2:])[0].bool()
        return feat, feat_mask

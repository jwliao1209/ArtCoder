from collections import namedtuple

import torch
from torch import nn
from torchvision.models import vgg16


class VGGFeatureExtractor(nn.Module):
    def __init__(
        self,
        requires_grad=False,
        pretrained_weights="DEFAULT",
    ):
        super(VGGFeatureExtractor, self).__init__()
        self.slices = [nn.Sequential() for _ in range(4)]
        self._initialize_slices(pretrained_weights)
        self.features = namedtuple("Outputs", [f"layer{i}" for i in range(4)])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def _initialize_slices(self, pretrained_weights):
        features = vgg16(weights=pretrained_weights).features 
        slice_indices = [(0, 4), (4, 9), (9, 16), (16, 23)]

        for slice_idx, (start, end) in enumerate(slice_indices):
            for i in range(start, end):
                self.slices[slice_idx].add_module(str(i), features[i])

        for param in self.parameters():
            param.requires_grad = False

    def to(self, device):
        super(VGGFeatureExtractor, self).to(device)
        self.slices = [s.to(device) for s in self.slices]
        return self

    def forward(self, x):
        outputs = []
        for i in range(4):
            x = self.slices[i](x)
            outputs.append(x)
        return self.features(*outputs)

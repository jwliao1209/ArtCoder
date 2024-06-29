from collections import namedtuple

import torch
from torch import nn
from torchvision.models import squeezenet1_1, vgg16


class BaseFeatureExtractor(nn.Module):
    def __init__(
        self,
        n_slices,
        requires_grad=False,
        pretrained_weights="DEFAULT",
    ):
        super(BaseFeatureExtractor, self).__init__()
        self.n_slices = n_slices
        self.slices = [nn.Sequential() for _ in range(self.n_slices)]
        self._initialize_slices(pretrained_weights)
        self.features = namedtuple("Outputs", [f"layer{i}" for i in range(self.n_slices)])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def _initialize_slices(self, pretrained_weights):
        raise NotImplementedError("Subclasses should implement this method.")

    def to(self, device):
        super(BaseFeatureExtractor, self).to(device)
        self.slices = [s.to(device) for s in self.slices]
        return self

    def forward(self, x):
        outputs = []
        for i in range(self.n_slices):
            x = self.slices[i](x)
            outputs.append(x)
        return self.features(*outputs)


class VGGFeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        requires_grad=False,
        pretrained_weights="DEFAULT",
    ):
        super(VGGFeatureExtractor, self).__init__(
            n_slices=4,
            requires_grad=requires_grad,
            pretrained_weights=pretrained_weights,
        )

    def _initialize_slices(self, pretrained_weights):
        features = vgg16(weights=pretrained_weights).features
        slice_indices = [(0, 4), (4, 9), (9, 16), (16, 23)]

        for slice_idx, (start, end) in enumerate(slice_indices):
            for i in range(start, end):
                self.slices[slice_idx].add_module(str(i), features[i])


class SqueezeFeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        requires_grad=False,
        pretrained_weights="DEFAULT",
    ):
        super(SqueezeFeatureExtractor, self).__init__(
            n_slices=7,
            requires_grad=requires_grad,
            pretrained_weights=pretrained_weights,
        )

    def _initialize_slices(self, pretrained_weights):
        features = squeezenet1_1(weights=pretrained_weights).features
        slice_indices = [(0, 2), (2, 5), (5, 8), (8, 10), (10, 11), (11, 12), (12, 13)]

        for slice_idx, (start, end) in enumerate(slice_indices):
            for i in range(start, end):
                self.slices[slice_idx].add_module(str(i), features[i])


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        model_type="vgg",
        pretrained_weights="DEFAULT",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(FeatureExtractor, self).__init__()

        if model_type == "vgg":
            extractor = VGGFeatureExtractor

        elif model_type == "squeeze":
            extractor = SqueezeFeatureExtractor

        self.extractor = extractor(
            pretrained_weights=pretrained_weights,
            requires_grad=False,
        ).to(device)
        
        self.device = device

    def forward(self, x):
        return self.extractor(x)

import math

import cv2
import torch
from torch import nn


class CenterFilter(nn.Module):
    def __init__(self, module_size: int):
        super(CenterFilter, self).__init__()
        self.module_size = module_size
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=module_size,
            stride=module_size,
            padding=0,
            bias=None,
            groups=1,
        )
        self.init_weights()

    def init_weights(self):
        module_center = int(self.module_size / 2) + 1
        center_filter = torch.zeros((1, 1, self.module_size, self.module_size))
        center_filter[:, :, module_center, module_center] = 1.0
        self.conv.weight = nn.Parameter(center_filter, requires_grad=False)

    @torch.no_grad()
    def forward(self, x):
        return self.conv(x)


class CenterMeanFilter(nn.Module):
    def __init__(self, module_size: int):
        super(CenterMeanFilter, self).__init__()
        self.module_size = module_size
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=module_size,
            stride=module_size,
            padding=0,
            bias=None,
            groups=1,
        )
        self.kernel_size = module_size
        self.init_weights()

    def init_weights(self):
        module_center = int(self.module_size / 2)
        radius = math.ceil(self.module_size / 6)
        center_filter = torch.zeros((1, 1, self.module_size, self.module_size))
        center_filter[
            :, :,
            module_center-radius : module_center+radius,
            module_center-radius : module_center+radius,
        ] = 1.0

        self.conv.weight = nn.Parameter(
            center_filter / center_filter.sum(),
            requires_grad=False,
        )

    @torch.no_grad()
    def forward(self, x):
        return self.conv(x)


class ErrorModuleFilter(nn.Module):
    def __init__(
        self,
        module_size: int,
        b_thres: float = 70 / 255,
        w_thres: float = 180 / 255
    ):
        super(ErrorModuleFilter, self).__init__()
        self.module_size = module_size
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=module_size,
            stride=module_size,
            padding=0,
            bias=None,
            groups=1,
        )
        self.center_mean_filter = CenterMeanFilter(module_size)
        self.module_size = module_size
        self.b_thres = b_thres
        self.w_thres = w_thres

    @torch.no_grad()
    def forward(self, x, y):
        x_center_mean = self.center_mean_filter(x)
        b_error = (y == 0) * (x_center_mean > self.b_thres)
        w_error = (y == 1) * (x_center_mean < self.w_thres)
        return b_error + w_error


class SamplingSimulationLayer(nn.Module):
    def __init__(self, module_size: int, filter_thres: float = 1e-3):
        super(SamplingSimulationLayer, self).__init__()
        self.module_size = module_size
        self.filter_thres = filter_thres
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=module_size,
            stride=module_size,
            padding=0,
            bias=False,
        )
        self.init_weights()

    def init_weights(self):
        sigma = int((self.module_size -1) / 5)
        filter_1d = cv2.getGaussianKernel(
            ksize=self.module_size,
            sigma=sigma,
            ktype=cv2.CV_32F,
        )
        filter_2d = torch.tensor(filter_1d * filter_1d.T, dtype=torch.float32)
        filter_2d[filter_2d < self.filter_thres] = .0
        gaussian_kernel_init = filter_2d.reshape(1, 1, *filter_2d.shape)
        self.conv.weight = nn.Parameter(gaussian_kernel_init, requires_grad=False)

    def forward(self, x):
        return self.conv(x)

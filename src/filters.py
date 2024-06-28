import math

import cv2
import torch
from torch import nn


class BinaryMeanFilter(nn.Module):
    def __init__(self, module_size, binary_thres=0.5):
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
        self.module_size = module_size
        self.binary_thres = binary_thres
        self.init_weights()

    def init_weights(self):
        ones_filter = torch.ones((1, 1, self.module_size, self.module_size))
        mean_filter = ones_filter / ones_filter.sum()
        self.conv.weight = nn.Parameter(mean_filter, requires_grad=False)

    @torch.no_grad()
    def forward(self, x):
        return (self.conv(x) >= self.binary_thres).float()


class CenterMeanFilter(nn.Module):
    def __init__(self, module_size):
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
    def __init__(self, module_size, b_thres=50, w_thres=200):
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
        y_center_mean = self.center_mean_filter(y)

        b_error = (y_center_mean == 0) & (x_center_mean < self.b_thres)
        w_error = (y_center_mean == 1) & (x_center_mean > self.w_thres)

        error_module = torch.ones_like(b_error, dtype=torch.float32, device=x.device)
        error_module[b_error | w_error] = .0

        return error_module


class SamplingSimulationLayer(nn.Module):
    def __init__(self, module_size):
        super(SamplingSimulationLayer, self).__init__()
        self.module_size = module_size
        self.conv = nn.Conv2d(
            in_channels=3,
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
            ktype=cv2.CV_32F
        )
        filter_2d = torch.tensor(filter_1d * filter_1d.T, dtype=torch.float32)
        gaussian_kernel_init = filter_2d.reshape(1, 1, *filter_2d.shape).repeat(1, 3, 1, 1)
        gaussian_kernel_init[gaussian_kernel_init < 0.1] = 0
        self.conv.weight = nn.Parameter(gaussian_kernel_init, requires_grad=False)

    def forward(self, x):
        return self.conv(x)
    

if __name__ == '__main__':
    ssl = SamplingSimulationLayer(module_size=24)
    x = torch.randn(1, 3, 240, 240)
    y = ssl(x)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)

    error_module_filter = ErrorModuleFilter(module_size=24)
    x = torch.randn(1, 1, 240, 240)
    y = (torch.randn(1, 1, 240, 240) > 0).float()
    z = error_module_filter(x, y)
    print("Input shape:", x.shape, y.shape)
    print("Output shape:", z.shape)

import torch
from torch import nn

from .feature_extractor import VGGFeatureExtractor
from .filters import ErrorModuleFilter, SamplingSimulationLayer, CenterFilter
from .image_processor import color_to_gray, image_binarize


class CodeLoss(nn.Module):
    def __init__(
        self,
        qrcode_image: torch.Tensor,
        module_size: int,
        b_thres: float,
        w_thres: float,
        b_soft_value: float,
        w_soft_value: float,
    ):
        super(CodeLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.error_module_filter = ErrorModuleFilter(
            module_size=module_size,
            b_thres=b_thres,
            w_thres=w_thres,
        )

        self.ss_layer = SamplingSimulationLayer(module_size=module_size)
        self.center_filter = CenterFilter(module_size=module_size)

        self.to(qrcode_image.device)
        self.qrcode_module_image = self.center_filter(image_binarize(qrcode_image))
        self.target_code = self.get_target_code(
            image_binarize(qrcode_image),
            b_soft_value,
            w_soft_value,
        )

    @staticmethod
    def get_target_code(qrcode_module_image, b_soft_value, w_soft_value):
        assert torch.equal(
            qrcode_module_image.unique(),
            torch.tensor([0., 1.], device=qrcode_module_image.device)
        )
        qrcode_module_image = qrcode_module_image.clone()
        qrcode_module_image[qrcode_module_image == 1] = w_soft_value
        qrcode_module_image[qrcode_module_image == 0] = b_soft_value
        return qrcode_module_image.repeat(1, 3, 1, 1)

    def forward(self, x):
        error_module = self.error_module_filter(
            color_to_gray(x.clone().detach()),
            self.qrcode_module_image,
        )
        return self.mse(
            self.ss_layer(x) * error_module,
            self.ss_layer(self.target_code) * error_module,
        )


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, fx: torch.Tensor, fy: torch.Tensor) -> torch.Tensor:
        return self.mse_loss(fx[-2], fy[-2])


class StyleFeatureLoss(nn.Module):
    def __init__(self):
        super(StyleFeatureLoss, self).__init__()
        self.mse = nn.MSELoss()

    @staticmethod
    def _gram_matrix(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        feature = x.view(b, c, h * w)
        return feature.bmm(feature.transpose(1, 2)) / (c * h * w)

    def forward(self, image_features: torch.Tensor, style_features: torch.Tensor) -> torch.Tensor:
        return sum(
            [
                self.mse(self._gram_matrix(fx), self._gram_matrix(fy))
                for fx, fy in zip(image_features, style_features)
            ]
        )


class ArtCoderLoss(nn.Module):
    def __init__(
        self,
        qrcode_image: torch.Tensor,
        content_image: torch.Tensor,
        style_image: torch.Tensor,
        module_size: int = 16,
        b_thres: float = 50,
        w_thres: float = 200,
        b_soft_value: float = 40 / 255,
        w_soft_value: float = 220 / 255,
        code_weight: float = 1e12,
        content_weight: float = 1e8,
        style_weight: float = 1e15,
        pretrained_weights: str = "DEFAULT",
        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(ArtCoderLoss, self).__init__()
        self.feature_extractor = VGGFeatureExtractor(
            pretrained_weights=pretrained_weights,
        ).to(device)
        self.code_loss = CodeLoss(
            qrcode_image=qrcode_image,
            module_size=module_size,
            b_thres=b_thres,
            w_thres=w_thres,
            b_soft_value=b_soft_value,
            w_soft_value=w_soft_value,
        )
        self.content_features = self.feature_extractor(content_image)
        self.style_features = self.feature_extractor(style_image)
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleFeatureLoss()
        self.code_weight = code_weight
        self.content_weight = content_weight
        self.style_weight = style_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        code_loss = self.code_loss(x)
        image_features = self.feature_extractor(x)
        perceptual_loss = self.perceptual_loss(image_features, self.content_features)
        style_loss = self.style_loss(image_features, self.style_features)
        total_loss = (
            self.code_weight * code_loss + \
            self.content_weight * perceptual_loss + \
            self.style_weight * style_loss
        )
        return {
            "code": code_loss,
            "perceptual": perceptual_loss,
            "style": style_loss,
            "total": total_loss
        }

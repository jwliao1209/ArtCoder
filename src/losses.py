import torch
from torch import nn

from .feature_extractor import VGGFeatureExtractor
from .filters import ErrorModuleFilter, SamplingSimulationLayer, CenterFilter
from .image_processor import color_to_gray, image_binarize


class CodeLoss(nn.Module):
    def __init__(
        self,
        module_size: int,
        b_thres: float,
        w_thres: float,
        b_soft_value: float,
        w_soft_value: float,
        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(CodeLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.error_module_filter = ErrorModuleFilter(
            module_size=module_size,
            b_thres=b_thres,
            w_thres=w_thres,
        )
        self.b_soft_value = b_soft_value
        self.w_soft_value = w_soft_value
        self.ss_layer = SamplingSimulationLayer(module_size=module_size)
        self.center_filter = CenterFilter(module_size=module_size)
        self.to(device)

    @staticmethod
    def _get_target_code(qrcode_module_image, b_soft_value, w_soft_value):
        assert torch.equal(
            qrcode_module_image.unique(),
            torch.tensor([0., 1.], device=qrcode_module_image.device)
        )
        qrcode_module_image = qrcode_module_image.clone()
        qrcode_module_image[qrcode_module_image == 1] = w_soft_value
        qrcode_module_image[qrcode_module_image == 0] = b_soft_value
        return qrcode_module_image.repeat(1, 3, 1, 1)
    
    def _get_error_module_mask(self, x: torch.Tensor, qrcode_image: torch.Tensor) -> torch.Tensor:
        qrcode_module_image = self.center_filter(image_binarize(qrcode_image))
        error_module_mask = self.error_module_filter(
            color_to_gray(x.clone().detach()),
            qrcode_module_image,
        )
        return error_module_mask

    def forward(self, x: torch.Tensor, qrcode_image: torch.Tensor) -> torch.Tensor:
        target_code = self._get_target_code(
            image_binarize(qrcode_image),
            self.b_soft_value,
            self.w_soft_value,
        )
        error_module_mask = self._get_error_module_mask(x, qrcode_image)
        return self.mse(
            self.ss_layer(x) * error_module_mask,
            self.ss_layer(target_code) * error_module_mask,
        )


class PerceptualLoss(nn.Module):
    def __init__(
        self,
        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(PerceptualLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.feature_extractor = VGGFeatureExtractor().to(device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.mse_loss(
            self.feature_extractor(x)[-2],
            self.feature_extractor(y)[-2],
        )


class StyleFeatureLoss(nn.Module):
    def __init__(
        self,
        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(StyleFeatureLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.feature_extractor = VGGFeatureExtractor().to(device)

    @staticmethod
    def _gram_matrix(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        feature = x.view(b, c, h * w)
        return feature.bmm(feature.transpose(1, 2)) / (c * h * w)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return sum(
            [
                self.mse(self._gram_matrix(fx), self._gram_matrix(fy))
                for fx, fy in zip(self.feature_extractor(x), self.feature_extractor(y))
            ]
        )


class ArtCoderLoss(nn.Module):
    def __init__(
        self,
        module_size: int = 16,
        b_thres: float = 50 / 255,
        w_thres: float = 200 / 255,
        b_soft_value: float = 40 / 255,
        w_soft_value: float = 220 / 255,
        code_weight: float = 1e12,
        content_weight: float = 1e8,
        style_weight: float = 1e15,
        pretrained_weights: str = "DEFAULT",
        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(ArtCoderLoss, self).__init__()
        self.code_loss = CodeLoss(
            module_size=module_size,
            b_thres=b_thres,
            w_thres=w_thres,
            b_soft_value=b_soft_value,
            w_soft_value=w_soft_value,
            device=device,
        )
        self.perceptual_loss = PerceptualLoss(device=device)
        self.style_loss = StyleFeatureLoss(device=device)
        self.code_weight = code_weight
        self.content_weight = content_weight
        self.style_weight = style_weight

    def forward(
        self,
        x: torch.Tensor,
        qrcode_image: torch.Tensor,
        content_image: torch.Tensor,
        style_image: torch.Tensor,
    ) -> torch.Tensor:

        code_loss = self.code_loss(x, qrcode_image)
        perceptual_loss = self.perceptual_loss(x, content_image)
        style_loss = self.style_loss(x, style_image)
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

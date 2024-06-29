import torch
from torch import nn

from .feature_extractor import FeatureExtractor
from .filters import ErrorModuleFilter, SamplingSimulationLayer, CenterFilter


def color_to_gray(images: torch.Tensor) -> torch.Tensor:
    assert images.shape[1] == 3, \
        f"The channel of color images must be 3 but get {images.shape[1]}. They are not color images."

    gray_image = 0.2999 * images[:, 0] + 0.587 * images[:, 1] + 0.1114 * images[:, 2]
    return gray_image.unsqueeze(1)


class CodeLoss(nn.Module):
    def __init__(
        self,
        qrcode_image: torch.Tensor,
        module_size: int,
        b_thres: float,
        w_thres:float ,
        b_soft_value: float,
        w_soft_value: float
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
        self.qrcode_module_image = self.center_filter(qrcode_image)
        self.target_code = self.get_target_code(
            self.qrcode_module_image.clone(),
            b_soft_value,
            w_soft_value,
        )

    @staticmethod
    def get_target_code(qrcode_module_image, b_soft_value, w_soft_value):
        qrcode_module_image[qrcode_module_image == 1] = w_soft_value
        qrcode_module_image[qrcode_module_image == 0] = b_soft_value
        return qrcode_module_image

    def forward(self, x):
        x_gray = color_to_gray(x)
        error_module = self.error_module_filter(
            x_gray.clone(),
            self.qrcode_module_image
        )

        return self.mse(
            self.ss_layer(x_gray) * error_module,
            self.target_code * error_module
        )


class GramMatrix(nn.Module):
    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        F = x.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        return G.div_(c * h * w)


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.gram = GramMatrix()
        self.mse = nn.MSELoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.mse(self.gram(x), self.gram(y))


class PerceptualLoss(nn.Module):
    def __init__(
        self,
        content_image: torch.Tensor,
        feature_extractor: FeatureExtractor,
        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(PerceptualLoss, self).__init__()
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.fy = feature_extractor(content_image)[-1]
        self.eval()

    def forward(self, fx: torch.Tensor) -> torch.Tensor:
        return self.mse_loss(fx, self.fy)


class StyleFeatureLoss(nn.Module):
    def __init__(
        self,
        style_image: torch.Tensor,
        feature_extractor: FeatureExtractor,
        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(StyleFeatureLoss, self).__init__()
        self.style_loss = StyleLoss()
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.style_image = style_image
        self.style_features = feature_extractor(style_image)
        self.eval()

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return sum(
            [
                self.style_loss(fx, fy)
                for fx, fy in zip(image_features, self.style_features)
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
        model_type: str = "vgg",
        pretrained_weights: str = "DEFAULT",
        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(ArtCoderLoss, self).__init__()
        self.device = device
        self.to(device)

        self.feature_extractor = FeatureExtractor(
            model_type=model_type,
            pretrained_weights=pretrained_weights,
            device=device,
        )

        self.code_loss = CodeLoss(
            qrcode_image=qrcode_image,
            module_size=module_size,
            b_thres=b_thres,
            w_thres=w_thres,
            b_soft_value=b_soft_value,
            w_soft_value=w_soft_value,
        )
        self.perceptual_loss = PerceptualLoss(
            content_image=content_image,
            feature_extractor=self.feature_extractor,
            device=device,
        )
        self.style_loss = StyleFeatureLoss(
            style_image=style_image,
            feature_extractor=self.feature_extractor,
            device=device,
        )
        self.code_weight = code_weight
        self.content_weight = content_weight
        self.style_weight = style_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image_features = self.feature_extractor(x)
        code_loss = self.code_loss(x)
        perceptual_loss = self.perceptual_loss(image_features[-1])
        style_loss = self.style_loss(image_features)
        total_loss = self.code_weight * code_loss + \
            self.content_weight * perceptual_loss + \
            self.style_weight * style_loss

        return {
            "code": code_loss,
            "perceptual": perceptual_loss,
            "style": style_loss,
            "total": total_loss
        }

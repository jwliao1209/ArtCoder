import torch
from torch import nn

from src.feature_extractor import FeatureExtractor
from src.filters import ErrorModuleFilter, SamplingSimulationLayer, CenterFilter


def color_to_gray(images):
    assert images.shape[1] == 3, \
        f"The channel of color images must be 3 but get {images.shape[1]}. They are not color images."

    gray_image = 0.2999 * images[:, 0] + 0.587 * images[:, 1] + 0.1114 * images[:, 2]
    return gray_image.unsqueeze(1)


class CodeLoss(nn.Module):
    def __init__(self, module_size, b_thres, w_thres):
        super(CodeLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.error_module_filter = ErrorModuleFilter(
            module_size=module_size,
            b_thres=b_thres,
            w_thres=w_thres,
        )
        self.ss_layer = SamplingSimulationLayer(module_size=module_size)
        self.center_filter = CenterFilter(module_size=module_size)

    def forward(self, x, y):
        x_gray = color_to_gray(x)
        error_module = self.error_module_filter(x_gray.clone(), y)
        return self.mse(
            self.ss_layer(x_gray) * error_module,
            self.center_filter(y) * error_module
        )


class GramMatrix(nn.Module):
    def forward(self, x):
        b, c, h, w = x.shape
        F = x.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        return G.div_(c * h * w)


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.gram = GramMatrix()
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        return self.mse(self.gram(x), self.gram(y))


class PerceptualLoss(nn.Module):
    def __init__(
        self,
        model_type="vgg",
        pretrained_weights="DEFAULT",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(PerceptualLoss, self).__init__()
        self.extractor = FeatureExtractor(
            model_type=model_type,
            pretrained_weights=pretrained_weights,
            device=device,
        )
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.eval()

    def forward(self, x, y):
        fx = self.extractor(x)[-1]
        fy = self.extractor(y)[-1]
        return self.mse_loss(fx, fy)


class StyleFeatureLoss(nn.Module):
    def __init__(
        self,
        model_type="vgg",
        pretrained_weights="DEFAULT",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(StyleFeatureLoss, self).__init__()
        self.style_loss = StyleLoss()
        self.extractor = FeatureExtractor(
            model_type=model_type,
            pretrained_weights=pretrained_weights,
            device=device,
        )
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.eval()

    def forward(self, x, y):
        return sum(
            [
                self.style_loss(fx, fy)
                for fx, fy in zip(self.extractor(x), self.extractor(y))
            ]
        )

class ArtCoderLoss(nn.Module):
    def __init__(
        self,
        module_size=24,
        b_thres=50,
        w_thres=200,
        code_weight=1e12,
        content_weight=1e8,
        style_weight=1e15,
        model_type="vgg",
        pretrained_weights="DEFAULT",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super(ArtCoderLoss, self).__init__()
        self.code_loss = CodeLoss(
            module_size=module_size,
            b_thres=b_thres,
            w_thres=w_thres,
        )
        self.perceptual_loss = PerceptualLoss(
            model_type=model_type,
            pretrained_weights=pretrained_weights,
            device=device,
        )
        self.style_loss = StyleFeatureLoss(
            model_type=model_type,
            pretrained_weights=pretrained_weights,
            device=device,
        )
        self.code_weight = code_weight
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.device = device

    def forward(self, x, y, x0, s):
        code_loss = self.code_loss(x, y)
        perceptual_loss = self.perceptual_loss(x, x0)
        style_loss = self.style_loss(x, s)
        #total_loss = self.code_weight * code_loss + \
        total_loss = self.content_weight * perceptual_loss + \
            self.style_weight * style_loss

        return {
            "code": code_loss,
            "perceptual": perceptual_loss,
            "style": style_loss,
            "total": total_loss
        }


if __name__ == "__main__":
    x = torch.randn((1, 3, 240, 240))
    y = torch.randn((1, 1, 240, 240))
    x0 = torch.randn((1, 3, 240, 240))
    s = torch.randn((1, 3, 240, 240))

    loss_func = ArtCoderLoss()
    loss = loss_func(x, y, x0, s)
    print(loss)

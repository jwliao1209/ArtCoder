import torch
from PIL import Image
from torch import nn
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from src.losses import ArtCoderLoss


def add_position_pattern(
        x: torch.Tensor,
        y: torch.Tensor,
        module_number: int,
        module_size: int
    ) -> torch.Tensor:

    x[: 8 * module_size - 1, : 8 * module_size - 1, :] = \
        y[: 8 * module_size - 1, : 8 * module_size - 1, :]

    x[
        (module_number - 8) * module_size + 1 : module_number * module_size,
        : 8 * module_size - 1,
        :
    ] = y[
        (module_number - 8) * module_size + 1 : module_number * module_size,
        : 8 * module_size - 1,
        :
    ]

    x[
        : 8 * module_size - 1,
        (module_number - 8) * module_size + 1 : module_number * module_size,
        :
    ] = y[
        : 8 * module_size - 1,
        (module_number - 8) * module_size + 1 : module_number * module_size,
        :
    ]

    x[
        (module_number - 9) * module_size : (module_number - 4) * module_size - 1,
        (module_number - 9) * module_size : (module_number - 4) * module_size - 1,
        :
    ] = y[
        (module_number - 9) * module_size : (module_number - 4) * module_size - 1,
        (module_number - 9) * module_size : (module_number - 4) * module_size - 1,
        :
    ]
    return x


def optimization(
    content_image: torch.Tensor,
    qrcode_image: torch.Tensor,
    style_image: torch.Tensor,
    module_size: int = 16,
    iterations: int = 50000,
    b_thres: float = 70 / 255,
    w_thres: float = 180 / 255,
    b_soft_value: float = 40 / 255,
    w_soft_value: float = 220 / 255,
    lr: float = 0.01,
    code_weight: float = 1e12,
    content_weight: float = 1e8,
    style_weight: float = 1e15,
):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_image = content_image.to(device)
    x = content_image.clone().requires_grad_(True)
    qrcode_image = qrcode_image[:, 0, :, :].unsqueeze(1).clone().requires_grad_(False).to(device)
    x0 = content_image.clone().requires_grad_(False).to(device)
    style_image = style_image.clone().requires_grad_(False).to(device)

    optimizer = torch.optim.Adam([x], lr=lr)
    objective_func = ArtCoderLoss(
        qrcode_image=qrcode_image,
        content_image=content_image,
        style_image=style_image,
        module_size=module_size,
        b_thres=b_thres,
        w_thres=w_thres,
        b_soft_value=b_soft_value,
        w_soft_value=w_soft_value,
        device=device,
        code_weight=code_weight,
        content_weight=content_weight,
        style_weight=style_weight,
    )

    for _ in tqdm(range(iterations)):
        def closure():
            optimizer.zero_grad()
            losses = objective_func(x)
            losses["total"].backward(retain_graph=True)
            print(", ".join([f"{k}: {v}" for k, v in losses.items()]))

        optimizer.step(closure)
    return x.clip(0, 1).squeeze().detach().permute(1, 2, 0).cpu().numpy() * 255, qrcode_image.clip(0, 1).squeeze(0).detach().permute(1, 2, 0).cpu().numpy() * 255


if __name__ == "__main__":
    content_image = Image.open("content/boy.jpg").resize((16 * 37, 16 * 37), Image.LANCZOS)
    qrcode_image = Image.open("code/boy.jpg").resize((16 * 37, 16 * 37), Image.LANCZOS)
    style_image = Image.open("style/texture1.1.jpg").resize((16 * 37, 16 * 37), Image.LANCZOS)

    content_image = pil_to_tensor(content_image).unsqueeze(0).float() / 255
    qrcode_image = ((pil_to_tensor(qrcode_image).unsqueeze(0).float() / 255) > 0.5).float()
    style_image = pil_to_tensor(style_image).unsqueeze(0).float() / 255

    x, y = optimization(
        content_image,
        qrcode_image,
        style_image,
        iterations=5000,
    )
    x = add_position_pattern(x, y, module_size=16, module_number=37)

    Image.fromarray(x.astype("uint8")).save("image.png")

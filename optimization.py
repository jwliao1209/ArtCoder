import torch
from PIL import Image
from torch import nn
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from src.losses import ArtCoderLoss


def optimization(
    content_image,
    qrcode_image,
    style_img,
    module_size=16,
    iterations=50000,
    b_thres=70 / 255,
    w_thres=180 / 255,
    lr=0.01,
    code_weight=1e9,
    content_weight=1e8,
    style_weight=1e15,
):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_image = content_image.to(device)
    # content_image = torch.zeros_like(content_image).to(device)
    x = content_image.clone().requires_grad_(True)
    y = qrcode_image[:, 0, :, :].unsqueeze(1).clone().requires_grad_(False).to(device)
    x0 = content_image.clone().requires_grad_(False).to(device)
    s = style_img.clone().requires_grad_(False).to(device)

    optimizer = torch.optim.Adam([x], lr=lr)
    objective_func = ArtCoderLoss(
        module_size=module_size,
        b_thres=b_thres,
        w_thres=w_thres,
        device=device,
        code_weight=code_weight,
        content_weight=content_weight,
        style_weight=style_weight,
    ).to(device)

    for _ in tqdm(range(iterations)):
        def closure():
            optimizer.zero_grad()
            losses = objective_func(x, y, x0, s)
            losses["total"].backward()

        optimizer.step(closure)
    return x.clip(0, 1).squeeze().detach().permute(1, 2, 0).cpu().numpy() * 255


if __name__ == "__main__":
    content_image = Image.open("content/boy.jpg").resize((16 * 37, 16 * 37), Image.LANCZOS)
    qrcode_image = Image.open("code/boy.jpg").resize((16 * 37, 16 * 37), Image.LANCZOS)
    style_image = Image.open("style/texture1.1.jpg").resize((16 * 37, 16 * 37), Image.LANCZOS)

    content_image = pil_to_tensor(content_image).unsqueeze(0).float() / 255
    qrcode_image = ((pil_to_tensor(qrcode_image).unsqueeze(0).float() / 255) > 0.5).float()
    style_image = pil_to_tensor(style_image).unsqueeze(0).float() / 255


    x = optimization(
        content_image,
        qrcode_image,
        style_image,
        iterations=1000,
    )

    print(content_image.shape)
    print(qrcode_image.shape)
    print(style_image.shape)
    Image.fromarray(x.astype("uint8")).save("image.png")

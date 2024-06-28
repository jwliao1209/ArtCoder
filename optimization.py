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
    module_size=37,
    iterations=50000,
    b_thres=70,
    w_thres=180,
    lr=0.01,
    code_weight=1e12,
    content_weight=1e8,
    style_weight=1e15,
):  
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    content_image = content_image.to(device)
    x = content_image.clone().requires_grad_(True)
    y = (qrcode_image[:, 0, :, :].unsqueeze(1).clone().requires_grad_(False).to(device) > 0.5).float()
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

    for i in tqdm(range(iterations)):
        def closure():
            optimizer.zero_grad()
            total_loss = objective_func(x, y, x0, s)
            total_loss.backward(retain_graph=True)
            return total_loss
        
        print(x.grad)

        optimizer.step(closure)
    return x.squeeze().detach().permute(1, 2, 0).cpu().numpy() * 255


if __name__ == "__main__":
    content_image = pil_to_tensor(Image.open("content/boy.jpg")).unsqueeze(0).float() / 255
    qrcode_image = pil_to_tensor(Image.open("code/boy.jpg")).unsqueeze(0).float() / 255
    style_image = pil_to_tensor(Image.open("style/texture1.1.jpg")).unsqueeze(0).float() / 255
    x = optimization(
        content_image,
        qrcode_image,
        style_image,
        iterations=10,
    )
    print(x.shape)
    Image.fromarray(x.astype("uint8")).show()

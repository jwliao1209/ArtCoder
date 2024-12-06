import os
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

from src.losses import ArtCoderLoss
from src.image_processor import image_binarize
from src.utils import (
    add_position_pattern,
    convert_normalized_tensor_to_np_image,
    convert_pil_to_normalized_tensor,
)


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--qrcode_image_path",
        type=str,
        default="images/code.jpg",
    )
    parser.add_argument(
        "--content_image_path",
        type=str,
        default="images/boy.jpg",
    )
    parser.add_argument(
        "--style_image_path",
        type=str,
        default="images/style.jpg",
    )
    parser.add_argument(
        "--module_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--module_num",
        type=int,
        default=37,
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--soft_black_value",
        type=float,
        default=40 / 255,
    )
    parser.add_argument(
        "--soft_white_value",
        type=float,
        default=220 / 255,
    )
    parser.add_argument(
        "--error_mask_black_thres",
        type=float,
        default=70 / 255,
    )
    parser.add_argument(
        "--error_mask_white_thres",
        type=float,
        default=180 / 255,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--code_weight",
        type=float,
        default=1e12,
    )
    parser.add_argument(
        "--content_weight",
        type=float,
        default=1e8,
    )
    parser.add_argument(
        "--style_weight",
        type=float,
        default=1e15,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/image.jpg",
    )
    return parser.parse_args()


def optimize_code(
    content_image: torch.Tensor,
    qrcode_image: torch.Tensor,
    style_image: torch.Tensor,
    module_size: int = 16,
    module_num: int = 37,
    iterations: int = 50000,
    soft_black_value: float = 40 / 255,
    soft_white_value: float = 220 / 255,
    error_mask_black_thres: float = 70 / 255,
    error_mask_white_thres: float = 180 / 255,
    lr: float = 0.01,
    code_weight: float = 1e12,
    content_weight: float = 1e8,
    style_weight: float = 1e15,
    display_loss: bool = True,
) -> np.array:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_image = content_image.to(device)
    qrcode_image = qrcode_image.to(device)
    style_image = style_image.to(device)

    x = content_image.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)
    objective_func = ArtCoderLoss(
        module_size=module_size,
        soft_black_value=soft_black_value,
        soft_white_value=soft_white_value,
        error_mask_black_thres=error_mask_black_thres,
        error_mask_white_thres=error_mask_white_thres,
        code_weight=code_weight,
        content_weight=content_weight,
        style_weight=style_weight,
        device=device,
    )

    for i in tqdm(range(iterations)):
        optimizer.zero_grad()
        losses = objective_func(x, qrcode_image, content_image, style_image)
        losses["total"].backward(retain_graph=True)
        optimizer.step()
        x.data.clamp_(0, 1)

        if display_loss:
            tqdm.write(
                f"iterations: {i}, " + \
                ", ".join([f"{k}_loss: {v:.4f}" for k, v in losses.items()])
            )

    return add_position_pattern(
        convert_normalized_tensor_to_np_image(x),
        convert_normalized_tensor_to_np_image(qrcode_image),
        module_size=module_size,
        module_num=module_num,
    )


if __name__ == "__main__":
    args = parse_arguments()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    qrcode_side_len = args.module_size * args.module_num
    qrcode_size = (qrcode_side_len, qrcode_side_len)

    content_image = Image.open(args.content_image_path).resize(qrcode_size, Image.LANCZOS)
    qrcode_image = Image.open(args.qrcode_image_path).resize(qrcode_size, Image.LANCZOS)
    style_image = Image.open(args.style_image_path).resize(qrcode_size, Image.LANCZOS)

    content_image = convert_pil_to_normalized_tensor(content_image)
    qrcode_image = convert_pil_to_normalized_tensor(qrcode_image)
    qrcode_image = image_binarize(qrcode_image)
    style_image = convert_pil_to_normalized_tensor(style_image)

    asethetic_qrcode = optimize_code(
        content_image,
        qrcode_image,
        style_image,
        module_size=args.module_size,
        module_num=args.module_num,
        iterations=args.iterations,
        soft_black_value=args.soft_black_value,
        soft_white_value=args.soft_white_value,
        error_mask_black_thres=args.error_mask_black_thres,
        error_mask_white_thres=args.error_mask_white_thres,
        code_weight=args.code_weight,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
    )
    asethetic_qrcode = to_pil_image(asethetic_qrcode)
    asethetic_qrcode.save(args.output_path)

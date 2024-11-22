# ArtCoder

This repository implements a PyTorch-based refactor of "ArtCoder: An End-to-end Method for Generating Scanning-robust Stylized QR Codes". The code has been restructured to enhance readability, scalability, and computational efficiency by leveraging PyTorch's built-in functions for tensor operations.


## Setup
To set up the environment:
1. Create a virtual environment:
```bash=
virtualenv --python=python3.10 artcoder
```bash=
2. Activate the environment:
```bash=
source artcoder/bin/activate
```
3. Install the dependencies:
```bash=
pip install -r requirements.txt
```


## Generation
To generate the aesthetic qrcode, run the following:
```bash=
python generate_aesthetic_qrcode.py \
    --qrcode_image_path <path_to_qrcode_image> \
    --content_image_path <path_to_content_image> \
    --style_image_path <path_to_style_image> \
    --output_path <path_to_output_image>
```


## Result
Below is an example of generating an aesthetic qrcode from an inputs:
<table>
    <tr>
        <td align="center">Original Image</td>
        <td align="center">Aesthetic QR Code</td> 
    </tr>
    <tr>
        <td height="250" width="280" align="center"><div align=center><img src="https://github.com/jwliao1209/Improved-ArtCoder/blob/main/images/boy.jpg" width="230" /></td>
        <td height="250" width="280" align="center"><div align=center><img src="https://github.com/jwliao1209/Improved-ArtCoder/blob/main/results/image.jpg" width="230" /></td>
    </tr>
</table>


## Environment
We implemented the code on an environment running Ubuntu 22.04.1, utilizing a 12th Generation Intel(R) Core(TM) i7-12700 CPU, along with a single NVIDIA GeForce RTX 4090 GPU equipped with 24 GB of dedicated memory.


## Citation
If you use this code, please cite the following repository:
```bibtex
@misc{liao2024artcoder,
    title  = {A PyTorch implementation of ArtCoder},
    author = {Jia-Wei Liao},
    url    = {https://github.com/jwliao1209/ArtCoder},
    year   = {2024}
}
```
Additionally, please reference the original paper:
```bibtex
@inproceedings{su2021artcoder,
  title     = {Artcoder: an end-to-end method for generating scanning-robust stylized qr codes},
  author    = {Su, Hao and Niu, Jianwei and Liu, Xuefeng and Li, Qingfeng and Wan, Ji and Xu, Mingliang and Ren, Tao},
  booktitle = {CVPR},
  year      = {2021}
}
```

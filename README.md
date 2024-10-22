# ArtCoder+

A fast PyTorch implementation of "ArtCoder: An End-to-end Method for Generating Scanning-robust Stylized QR Codes".


## Setup
```
virtualenv --python=python3.10 artcoder
source diffqrcoder/bin/activate
pip install -r requirements.txt
```

## Generation
```
python generate_asethetic_qrcode.py
```

## Result
<table>
    <tr>
        <td align="center">Original Image</td>
        <td align="center">Aesthetic QR Code</td> 
    </tr>
    <tr>
        <td height="250" width="280" align="center"><div align=center><img src="https://github.com/jwliao1209/Improved-ArtCoder/blob/main/images/boy.jpg" width="230" /></td>
        <td rowspan="2" align="center" width="550" height="550"><img src="https://github.com/jwliao1209/Improved-ArtCoder/blob/main/results/image.jpg" width="500" /></td>
    </tr>
</table>


## Environment
We implemented the code on an environment running Ubuntu 22.04.1, utilizing a 12th Generation Intel(R) Core(TM) i7-12700 CPU, along with a single NVIDIA GeForce RTX 4090 GPU equipped with 24 GB of dedicated memory.


## Citation
```
@misc{source_separation_2024,
    title  = {Improved-ArtCoder: a fast PyTorch implementation of ArtCoder},
    author = {Jia-Wei Liao},
    url    = {https://github.com/jwliao1209/Improved-ArtCoder},
    year   = {2024}
}
```

```
@inproceedings{su2021artcoder,
  title={Artcoder: an end-to-end method for generating scanning-robust stylized qr codes},
  author={Su, Hao and Niu, Jianwei and Liu, Xuefeng and Li, Qingfeng and Wan, Ji and Xu, Mingliang and Ren, Tao},
  booktitle={CVPR},
  year={2021}
}
```

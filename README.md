# StarKAN: A Lightweight Frequency-Aware Framework for Crack Segmentation in Geotechnical and Infrastructure Monitoring

This repository contains the official PyTorch implementation of **StarKAN**, as described in the paper:

> Qin, H., Peng, F., Luo, X., Hu, Z., Jiang, N. "A Lightweight Frequency-Aware Framework for Crack Segmentation in Geotechnical and Infrastructure Monitoring." *Computers & Geosciences* (under review).

---

## Overview

StarKAN is a lightweight crack segmentation framework designed for deployment in geotechnical field monitoring scenarios, where image quality and computational resources are constrained. The framework integrates:

- **LWGA-WT Encoder**: Spectral-spatial feature extraction with Haar-based Wavelet Transform Convolution (WTConv)
- **Neural Kolmogorov Mixer (NKM)**: RBF-based nonlinear bottleneck inspired by Kolmogorov-Arnold Networks
- **HyperUp Decoder**: Multi-scale Spectral Injection (MSI) and Edge-Gated Attention (EGA) for topology-preserving reconstruction
- **Soft-CLDice Loss**: Topology-aware training objective for structural continuity

**Key results on DeepCrack:** 88.47% mIoU, 90.89% clDice, 2.66M parameters, 1.10 GFLOPs.

---

## Requirements

```
Python >= 3.8
PyTorch >= 2.0
torchvision
numpy
tqdm
albumentations
```

Install dependencies:

```bash
pip install torch torchvision numpy tqdm albumentations
```

---

## Dataset Preparation

Download the datasets and organize as follows:

```
DeepCrack/
  train/
    images/
    masks/
  test/
    images/
    masks/
```

- **DeepCrack**: [https://github.com/yhlleo/DeepCrack](https://github.com/yhlleo/DeepCrack)
- **CFD**: Shi et al., IEEE T-ITS, 2016
- **Crack500**: Yang et al., IEEE T-ITS, 2020

---

## Training

```bash
python train.py
```

Key configuration options in `train.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `root_dir` | `./DeepCrack` | Dataset root directory |
| `img_size` | `320` | Input patch size |
| `batch_size` | `8` | Training batch size |
| `epochs` | `120` | Total training epochs |
| `lr` | `1e-3` | Initial learning rate |
| `device` | auto | `mps` / `cuda` / `cpu` |

The training script automatically selects the available device (MPS for Apple Silicon, CUDA for NVIDIA GPU, or CPU).

---

## Inference

```python
import torch
from model import StarKAN

model = StarKAN(n_classes=1)
model.load_state_dict(torch.load('checkpoints/StarKAN_DeepCrack/best_model.pth'))
model.eval()

# Input: [B, 3, H, W] normalized image tensor
# Output: [B, 1, H, W] crack probability map
```

---

## Model Architecture

```
Input (3 x H x W)
  |
  Fine-Grained Stem (2x 3x3 conv, stride 2)
  |
  LWGA-WT Stages (x4) -- WTConv replaces standard local aggregation
  |
  Neural Kolmogorov Mixer (NKM) -- RBF-based nonlinear bottleneck
  |
  HyperUp Decoder (x4) -- MSI + EGA + Gated Local Mixer
  |
Output (1 x H x W)
```

---

## Computational Efficiency

Benchmarked on Apple Mac mini (M4, 16GB RAM), PyTorch 2.6 with MPS acceleration, input size 320×320:

| Model | Params | GFLOPs | FPS |
|-------|--------|--------|-----|
| StarKAN (ours) | 2.66M | 1.10G | 23.24 |
| MambaOut-Femto | 6.48M | 6.19G | 6.48 |
| SegFormer-B0 | 3.71M | 2.63G | 21.2 |

---

## License

This project is released under the MIT License.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{qin2025starkan,
  title={A Lightweight Frequency-Aware Framework for Crack Segmentation in Geotechnical and Infrastructure Monitoring},
  author={Qin, Haishan and Peng, Fei and Luo, Xuanli and Hu, Zihao and Jiang, Ningjun},
  journal={Computers \& Geosciences},
  year={2025},
  note={under review}
}
```

---

## Acknowledgements

This work was supported by the Natural Science Foundation of China (Grant No. 42377166).  
The LWGA-WT backbone builds on [LWGANet](https://arxiv.org/abs/2501.10040) (Lu et al., 2025).

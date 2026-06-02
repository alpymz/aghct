# AGHCT: Attention-Guided Hybrid CNN-Transformer for Medical Image Segmentation

This repository contains the official implementation of **AGHCT**, an Attention-Guided Hybrid CNN-Transformer architecture for medical image segmentation in low-data regimes. The project was developed as a term project for the Digital Image Processing course at Abdullah Gül University.

## Overview

Medical image segmentation faces two main challenges: scarcity of annotated training data and the need to capture both local boundary details and global contextual information. AGHCT addresses both through:

- Lightweight CNN encoder with depthwise separable convolutions
- Dual-Attention Gate Module (DAGM) combining channel-wise (SE) and spatial (Local Window Transformer) attention
- Learnable gating that adaptively balances local and global features
- 22% fewer parameters than standard U-Net while matching its performance

## Results

Evaluated on the ISIC 2018 Skin Lesion Segmentation Challenge dataset using 2-fold cross-validation:

| Model | Parameters | Dice (mean ± std) | IoU | Sensitivity | Specificity |
|-------|-----------|-------------------|-----|-------------|-------------|
| U-Net | 31.04M | 0.8977 ± 0.0035 | 0.8124 | 0.9010 | 0.9707 |
| **AGHCT (Ours)** | **24.20M (-22%)** | **0.8973 ± 0.0003** | 0.8117 | 0.8915 | 0.9696 |

AGHCT achieves comparable performance with significantly fewer parameters and lower variance across folds.

## Repository Structure

```
aghct/
├── configs/
│   └── config.yaml
├── data/
│   ├── isic_dataset.py
│   ├── drive_dataset.py
│   └── augmentations.py
├── models/
│   ├── cnn_encoder.py
│   ├── dagm.py
│   ├── decoder.py
│   ├── aghct.py
│   ├── unet.py
│   └── transunet.py
├── losses/
│   └── losses.py
├── pretrain/
│   └── simclr.py
├── utils/
│   ├── metrics.py
│   └── visualization.py
├── train.py
├── evaluate.py
├── pretrain_simclr.py
└── requirements.txt
```

## Installation

Requires Python 3.8+ and CUDA-capable GPU.

```bash
git clone https://github.com/alpymz/aghct.git
cd aghct
pip install -r requirements.txt
```

## Usage

### Option 1: Google Colab (Recommended)

1. Upload the `aghct/` folder to your Google Drive
2. Upload the ISIC 2018 dataset zip to your Drive
3. Open `notebooks/AGHCT_Training.ipynb` in Google Colab
4. Select GPU runtime (Runtime → Change runtime type → T4 GPU)
5. Run cells sequentially

### Option 2: Command Line

```bash
# Train U-Net baseline
python train.py --config configs/config.yaml --model unet --dataset isic --fold 0 --epochs 50

# Train AGHCT
python train.py --config configs/config.yaml --model aghct --dataset isic --fold 0 --epochs 50

# Resume from checkpoint
python train.py --config configs/config.yaml --model aghct --dataset isic --fold 0 --epochs 50 --resume

# Evaluate
python evaluate.py --config configs/config.yaml --model aghct --checkpoint checkpoints/aghct_isic_fold0_best.pth
```

## Dataset

Download the ISIC 2018 Task 1 (Segmentation) dataset from Kaggle:

https://www.kaggle.com/datasets/tschandl/isic2018-challenge-task1-data-segmentation

## Architecture

The DAGM module combines two complementary attention branches.

**SE Branch (channel attention)**

- Global average pooling produces channel descriptor z
- Two-layer bottleneck (reduction ratio r=16) produces channel weights
- Output: F_SE = s ⊙ F

**LWT Branch (spatial attention)**

- Partitions feature map into 8×8 non-overlapping windows
- Multi-head self-attention within each window with relative positional bias
- 4 Transformer layers with 8 attention heads

**Gated Fusion**

- α = σ(Conv1x1([F_SE ; F_LWT]))
- F̂ = α ⊙ F_SE + (1-α) ⊙ F_LWT

## Training Configuration

- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Schedule: Cosine annealing with 10-epoch linear warm-up
- Loss: BCE + Dice (equal weight)
- Epochs: 50
- Batch size: 16
- Image size: 256×256
- Mixed precision (AMP) enabled
- Hardware: NVIDIA Tesla T4 (16 GB VRAM) via Google Colab

## Limitations and Future Work

This implementation does not yet include:

- SimCLR contrastive pre-training (designed but not evaluated due to compute constraints)
- Low-data regime experiments (10%, 25%, 50% training fractions)
- DRIVE retinal vessel dataset evaluation
- Ablation studies on DAGM components

## Citation

```bibtex
@misc{yilmaz2026aghct,
  author = {Yılmaz, Alperen},
  title = {Attention-Guided Hybrid CNN-Transformer Architecture for Medical Image Segmentation in Low-Data Regimes},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/alpymz/aghct}}
}
```

## Acknowledgements

This work was developed as a term project for the Digital Image Processing course at Abdullah Gül University. AI tools (Claude, Anthropic) were used for code implementation assistance, debugging, and manuscript preparation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Alperen Yılmaz - Department of Computer Engineering, Abdullah Gül University

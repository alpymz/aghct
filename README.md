AGHCT: Attention-Guided Hybrid CNN-Transformer for Medical Image Segmentation
This repository contains the official implementation of AGHCT, an Attention-Guided Hybrid CNN-Transformer architecture for medical image segmentation in low-data regimes. The project was developed as a term project for the Digital Image Processing course at Abdullah Gül University.
Overview
Medical image segmentation faces two main challenges: scarcity of annotated training data and the need to capture both local boundary details and global contextual information. AGHCT addresses both through:
Lightweight CNN encoder with depthwise separable convolutions
Dual-Attention Gate Module (DAGM) combining channel-wise (SE) and spatial (Local Window Transformer) attention
Learnable gating that adaptively balances local and global features
22% fewer parameters than standard U-Net while matching its performance
Results
Evaluated on the ISIC 2018 Skin Lesion Segmentation Challenge dataset using 2-fold cross-validation:
Model	Parameters	Dice (mean ± std)	IoU	Sensitivity	Specificity
U-Net	31.04M	0.8977 ± 0.0035	0.8124	0.9010	0.9707
AGHCT (Ours)	24.20M (-22%)	0.8973 ± 0.0003	0.8117	0.8915	0.9696
AGHCT achieves comparable performance with significantly fewer parameters and lower variance across folds.
Repository Structure
```
aghct/
├── configs/
│   └── config.yaml              # Hyperparameters and paths
├── data/
│   ├── isic_dataset.py          # ISIC 2018 dataset loader
│   ├── drive_dataset.py         # DRIVE retinal vessel dataset loader
│   └── augmentations.py         # Albumentations pipeline
├── models/
│   ├── cnn_encoder.py           # Depthwise separable CNN encoder
│   ├── dagm.py                  # Dual-Attention Gate Module
│   ├── decoder.py               # FPN-style feature fusion decoder
│   ├── aghct.py                 # Main AGHCT model
│   ├── unet.py                  # U-Net baseline
│   └── transunet.py             # TransUNet baseline
├── losses/
│   └── losses.py                # BCE + Dice combined loss
├── pretrain/
│   └── simclr.py                # SimCLR contrastive pre-training
├── utils/
│   ├── metrics.py               # Dice, IoU, Sensitivity, Specificity, HD95
│   └── visualization.py         # Attention map visualization
├── notebooks/
│   └── AGHCT_Training.ipynb     # Google Colab training notebook
├── train.py                     # Main training entry point
├── evaluate.py                  # Evaluation script
├── pretrain_simclr.py           # Contrastive pre-training entry point
├── collect_results.py           # Results aggregation
├── results/
│   ├── qualitative_results.png  # Sample predictions
│   └── results.txt              # Final metrics
└── requirements.txt
```
Installation
Requires Python 3.8+ and CUDA-capable GPU.
```bash
git clone https://github.com/alpymz/aghct.git
cd aghct
pip install -r requirements.txt
```
Usage
Option 1: Google Colab (Recommended)
The easiest way to reproduce results is via the included Colab notebook:
Upload the `aghct/` folder to your Google Drive
Upload the ISIC 2018 dataset zip to your Drive
Open `notebooks/AGHCT_Training.ipynb` in Google Colab
Select GPU runtime (Runtime → Change runtime type → T4 GPU)
Run cells sequentially
The notebook handles dataset extraction, model training, evaluation, and visualization automatically. Checkpoints are saved to Google Drive, allowing training to be resumed across multiple sessions.
Option 2: Command Line
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
Dataset
Download the ISIC 2018 Task 1 (Segmentation) dataset from:
Kaggle (recommended)
Official ISIC Archive
Expected directory structure:
```
isic2018/
├── ISIC2018_Task1-2_Training_Input/    # Training images
├── ISIC2018_Task1_Training_GroundTruth/ # Training masks
├── ISIC2018_Task1-2_Validation_Input/   # Validation images
└── ISIC2018_Task1-2_Test_Input/         # Test images
```
Architecture
The DAGM module combines two complementary attention branches:
SE Branch (channel attention):
Global average pooling produces channel descriptor z
Two-layer bottleneck (reduction ratio r=16) produces channel weights s = σ(W₂ · ReLU(W₁ · z))
Output: F_SE = s ⊙ F
LWT Branch (spatial attention):
Partitions feature map into 8×8 non-overlapping windows
Applies multi-head self-attention within each window with relative positional bias
4 Transformer layers with 8 attention heads
Gated Fusion:
α = σ(Conv₁ₓ₁([F_SE ; F_LWT]))
F̂ = α ⊙ F_SE + (1-α) ⊙ F_LWT
Training Configuration
Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
Schedule: Cosine annealing with 10-epoch linear warm-up
Loss: BCE + Dice (equal weight)
Epochs: 50
Batch size: 16
Image size: 256×256
Mixed precision (AMP) enabled
Hardware: NVIDIA Tesla T4 (16 GB VRAM) via Google Colab
Limitations and Future Work
This implementation does not yet include:
SimCLR contrastive pre-training (designed but not evaluated due to compute constraints)
Low-data regime experiments (10%, 25%, 50% training fractions)
DRIVE retinal vessel dataset evaluation
Ablation studies on DAGM components
These are planned for future work and are expected to demonstrate the architecture's advantages in low-data scenarios more clearly.
Citation
If you use this code in your research, please cite:
```bibtex
@misc{yilmaz2026aghct,
  author = {Yılmaz, Alperen},
  title = {Attention-Guided Hybrid CNN-Transformer Architecture for Medical Image Segmentation in Low-Data Regimes},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/alpymz/aghct}}
}
```
Acknowledgements
This work was developed as a term project for the Digital Image Processing course at Abdullah Gül University. AI tools (Claude, Anthropic) were used for code implementation assistance, debugging, and manuscript preparation.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Contact
Alperen Yılmaz - Department of Computer Engineering, Abdullah Gül University

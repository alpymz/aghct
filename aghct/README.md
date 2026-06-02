# AGHCT — Attention-Guided Hybrid CNN-Transformer

Medical image segmentation with a hybrid CNN-Transformer architecture (Dual-Attention
Gate Module) and contrastive SimCLR pre-training. Designed to run on Kaggle Notebooks
(T4 GPU, 16 GB VRAM) with full session-resume support.

## Structure

```
aghct/
├── configs/config.yaml
├── data/         # ISIC + DRIVE datasets, augmentations
├── models/       # CNN encoder, DAGM, decoder, AGHCT + baselines
├── losses/       # BCE + Dice loss
├── pretrain/     # SimCLR pre-training
├── utils/        # metrics, visualization
├── train.py
├── evaluate.py
├── pretrain_simclr.py
├── collect_results.py
└── notebooks/    # Kaggle entry-point scripts
```

## Quick start (Kaggle)

```bash
!pip install -q albumentations medpy einops timm

# 1. SimCLR pre-training (~4 h)
!python pretrain_simclr.py --dataset isic --config configs/config.yaml

# 2. Training (resumable)
!python train.py --config configs/config.yaml --model aghct --dataset isic --fold 0
!python train.py --config configs/config.yaml --model aghct --dataset isic --fold 0 --resume

# 3. Evaluation
!python evaluate.py --config configs/config.yaml --model aghct \
    --checkpoint /kaggle/working/checkpoints/aghct_isic_fold0_frac1.0_best.pth \
    --dataset isic
```

## Paths (Kaggle)

| What            | Where                                  |
|-----------------|----------------------------------------|
| Input datasets  | `/kaggle/input/`                       |
| Project code    | `/kaggle/input/aghct-code/` (uploaded) |
| Checkpoints     | `/kaggle/working/checkpoints/`         |
| Logs            | `/kaggle/working/logs/`                |
| Results         | `/kaggle/working/results/`             |

See `notebooks/` for ready-to-run Kaggle scripts.

# ===========================================================================
# Kaggle Notebook #1 — SimCLR Pre-training
# GPU: T4 (16 GB) | Tahmini süre: ~3-4 saat (ISIC) / ~2 saat (DRIVE)
# ===========================================================================
# Kaggle Dataset olarak ekle:
#   1) isic-2018                 (görüntüler)
#   2) aghct-code               (bu proje, aghct/ klasörü)
# Internet: KAPALI    GPU: AÇIK
# ---------------------------------------------------------------------------

# %% [code]
import os, sys, shutil, subprocess

# 1. pip — Kaggle imajında çoğu kütüphane var ama bunlar olmayabilir
subprocess.run(
    ["pip", "install", "-q", "albumentations", "medpy", "einops", "timm"],
    check=False,
)

# 2. Proje kodlarını çalışma dizinine kopyala (path'ler düzgün çalışsın)
CODE_SRC = "/kaggle/input/aghct-code/aghct"   # yüklenen dataset adı
CODE_DST = "/kaggle/working/aghct"
if not os.path.isdir(CODE_DST):
    shutil.copytree(CODE_SRC, CODE_DST)

os.chdir(CODE_DST)
sys.path.insert(0, CODE_DST)
print("CWD =", os.getcwd())
print("Files:", os.listdir(CODE_DST)[:10])


# %% [code]
# 3. ISIC SimCLR pre-training (~4 saat, 100 epoch, batch 128)
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# İlk session
subprocess.run([
    "python", "pretrain_simclr.py",
    "--dataset", "isic",
    "--config", "configs/config.yaml",
], check=True)


# %% [code]
# 4. Session kesilirse — bir sonraki session'da bu hücreyi çalıştır
# subprocess.run([
#     "python", "pretrain_simclr.py",
#     "--dataset", "isic",
#     "--config", "configs/config.yaml",
#     "--resume",
# ], check=True)


# %% [code]
# 5. DRIVE Pre-training (opsiyonel, ~2 saat)
# subprocess.run([
#     "python", "pretrain_simclr.py",
#     "--dataset", "drive",
#     "--config", "configs/config.yaml",
# ], check=True)


# %% [code]
# 6. Sonuçları doğrula — bu dosyaları yeni bir Kaggle Dataset olarak kaydet
ckpt_dir = "/kaggle/working/checkpoints"
print("Checkpoint dir contents:")
for f in sorted(os.listdir(ckpt_dir)):
    size_mb = os.path.getsize(os.path.join(ckpt_dir, f)) / (1024 * 1024)
    print(f"  {f}  ({size_mb:.1f} MB)")
# >> Output sekmesinden "Save as Dataset" → adı: aghct-pretrained

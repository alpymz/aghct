# ===========================================================================
# Kaggle Notebook #3 — AGHCT Training (all experiments)
# GPU: T4 16GB
# ===========================================================================
# Datasets:
#   1) isic-2018
#   2) aghct-code
#   3) aghct-pretrained  (Notebook #1'in çıktısı — SimCLR encoder checkpoint'ları)
# ---------------------------------------------------------------------------

# %% [code]
import os, sys, shutil, subprocess

subprocess.run(
    ["pip", "install", "-q", "albumentations", "medpy", "einops", "timm"],
    check=False,
)

CODE_SRC = "/kaggle/input/aghct-code/aghct"
CODE_DST = "/kaggle/working/aghct"
if not os.path.isdir(CODE_DST):
    shutil.copytree(CODE_SRC, CODE_DST)
os.chdir(CODE_DST)
sys.path.insert(0, CODE_DST)
print("CWD =", os.getcwd())


# %% [code]
# Pre-trained encoder dosyalarını working/checkpoints'e taşı
pretrain_src = "/kaggle/input/aghct-pretrained/checkpoints"
ckpt_dst = "/kaggle/working/checkpoints"
os.makedirs(ckpt_dst, exist_ok=True)
if os.path.isdir(pretrain_src):
    for f in os.listdir(pretrain_src):
        src = os.path.join(pretrain_src, f)
        dst = os.path.join(ckpt_dst, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    print("Pretrain checkpoints copied:", os.listdir(ckpt_dst))
else:
    print(f"WARN: {pretrain_src} not found — AGHCT will train from scratch")


# %% [markdown]
# ## Plan (each session ~12 hours)
#
# | Session | What                                              |
# |---------|---------------------------------------------------|
# | 1       | AGHCT full data, fold 0, 1                        |
# | 2       | AGHCT full data, fold 2, 3, 4                     |
# | 3       | AGHCT low-data (10%, 25%) × fold 0, 1, 2          |
# | 4       | AGHCT low-data (50%) × fold 0, 1, 2, 3, 4         |
# | 5       | Ablations: no_pretrain, SE-only, LWT-only         |


# %% [code]
def run_train(model, dataset, fold, fraction=1.0, resume=False, no_pretrain=False, epochs=None):
    cmd = [
        "python", "train.py",
        "--config", "configs/config.yaml",
        "--model", model,
        "--dataset", dataset,
        "--fold", str(fold),
        "--fraction", str(fraction),
    ]
    if resume:
        cmd.append("--resume")
    if no_pretrain:
        cmd.append("--no-pretrain")
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


# %% [code]
# === Session 1: AGHCT full data, fold 0, 1 ===
run_train("aghct", "isic", fold=0)
run_train("aghct", "isic", fold=1)


# %% [code]
# === Session 2: AGHCT full data, fold 2, 3, 4 ===
# run_train("aghct", "isic", fold=2)
# run_train("aghct", "isic", fold=3)
# run_train("aghct", "isic", fold=4)


# %% [code]
# === Session 3: Low-data, fraction 10% and 25% ===
# for frac in (0.10, 0.25):
#     for fold in (0, 1, 2):
#         run_train("aghct", "isic", fold=fold, fraction=frac)


# %% [code]
# === Session 4: Low-data, fraction 50% ===
# for fold in (0, 1, 2, 3, 4):
#     run_train("aghct", "isic", fold=fold, fraction=0.50)


# %% [code]
# === Session 5: Ablation studies ===
# AGHCT without SimCLR pre-training
# run_train("aghct", "isic", fold=0, no_pretrain=True)

# NOTE: SE-only / LWT-only ablation runs require dedicated configs.
# Edit ``configs/config.yaml`` (e.g. set ``model.dagm.num_transformer_layers: 0``
# for SE-only or skip the SE branch) and run again.


# %% [code]
# === List checkpoints to verify ===
for f in sorted(os.listdir(ckpt_dst)):
    size_mb = os.path.getsize(os.path.join(ckpt_dst, f)) / (1024 * 1024)
    print(f"  {f}  ({size_mb:.1f} MB)")
# >> "Save as Dataset" → adı: aghct-checkpoints-aghct

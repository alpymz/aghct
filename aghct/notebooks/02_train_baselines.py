# ===========================================================================
# Kaggle Notebook #2 — Baseline Trainings (U-Net + TransUNet)
# GPU: T4 16GB | Her model fold başına ~2-3 saat
# ===========================================================================
# Datasets:
#   1) isic-2018
#   2) aghct-code
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


# %% [markdown]
# ## Plan (split across 12-hour sessions)
#
# | Session | What                                |
# |---------|-------------------------------------|
# | 1       | U-Net fold 0, 1                     |
# | 2       | U-Net fold 2, 3, 4                  |
# | 3       | TransUNet fold 0, 1                 |
# | 4       | TransUNet fold 2, 3, 4              |
#
# Always pass ``--resume`` if a previous session was interrupted.


# %% [code]
def run_train(model, dataset, fold, fraction=1.0, resume=False, no_pretrain=False):
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
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


# %% [code]
# === Session 1: U-Net fold 0, 1 ===
run_train("unet", "isic", fold=0)
run_train("unet", "isic", fold=1)


# %% [code]
# === Session 2 (after dataset upload): U-Net fold 2, 3, 4 ===
# run_train("unet", "isic", fold=2)
# run_train("unet", "isic", fold=3)
# run_train("unet", "isic", fold=4)


# %% [code]
# === Session 3: TransUNet fold 0, 1 ===
# run_train("transunet", "isic", fold=0)
# run_train("transunet", "isic", fold=1)


# %% [code]
# === Session 4: TransUNet fold 2, 3, 4 ===
# run_train("transunet", "isic", fold=2)
# run_train("transunet", "isic", fold=3)
# run_train("transunet", "isic", fold=4)


# %% [code]
# === Resume any interrupted run ===
# run_train("unet", "isic", fold=1, resume=True)


# %% [code]
# === List checkpoints to verify ===
ckpt_dir = "/kaggle/working/checkpoints"
for f in sorted(os.listdir(ckpt_dir)):
    size_mb = os.path.getsize(os.path.join(ckpt_dir, f)) / (1024 * 1024)
    print(f"  {f}  ({size_mb:.1f} MB)")
# >> "Save as Dataset" → adı: aghct-checkpoints-baseline

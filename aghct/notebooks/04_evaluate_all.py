# ===========================================================================
# Kaggle Notebook #4 — Evaluate everything + result tables
# GPU: T4 (evaluation only — ~1-2 hours)
# ===========================================================================
# Datasets:
#   1) isic-2018
#   2) aghct-code
#   3) aghct-checkpoints           (all trained models)
# ---------------------------------------------------------------------------

# %% [code]
import os, sys, shutil, subprocess, json, glob, re

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


# %% [code]
# Bring all "best" checkpoints to /kaggle/working/checkpoints
ckpt_src = "/kaggle/input/aghct-checkpoints/checkpoints"
ckpt_dst = "/kaggle/working/checkpoints"
os.makedirs(ckpt_dst, exist_ok=True)
if os.path.isdir(ckpt_src):
    for f in os.listdir(ckpt_src):
        if f.endswith(".pth"):
            dst = os.path.join(ckpt_dst, f)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(ckpt_src, f), dst)
print("Local checkpoints:", sorted(os.listdir(ckpt_dst)))


# %% [code]
# Parse checkpoint filenames into (model, dataset, fold, fraction, suffix)
CKPT_RE = re.compile(
    r"^(?P<model>aghct|unet|transunet)_"
    r"(?P<dataset>isic|drive)_"
    r"fold(?P<fold>\d+)_"
    r"frac(?P<frac>[\d.]+)"
    r"(?P<suffix>(_nopretrain)?)_best\.pth$"
)

def parse(fname):
    m = CKPT_RE.match(fname)
    if not m:
        return None
    return {
        "model": m["model"],
        "dataset": m["dataset"],
        "fold": int(m["fold"]),
        "fraction": float(m["frac"]),
        "suffix": m["suffix"] or "",
    }

ckpts = []
for f in sorted(os.listdir(ckpt_dst)):
    info = parse(f)
    if info:
        info["path"] = os.path.join(ckpt_dst, f)
        ckpts.append(info)
print(f"Found {len(ckpts)} `_best.pth` checkpoints")


# %% [code]
# Run evaluate.py on each and store metric JSONs
results_dir = "/kaggle/working/results"
os.makedirs(results_dir, exist_ok=True)
for c in ckpts:
    print(f"\n=== {os.path.basename(c['path'])} ===")
    subprocess.run([
        "python", "evaluate.py",
        "--config", "configs/config.yaml",
        "--model", c["model"],
        "--dataset", c["dataset"],
        "--checkpoint", c["path"],
        "--fold", str(c["fold"]),
        "--results-dir", results_dir,
        "--no-qualitative",   # speed: enable only if needed
        "--no-attention",
    ], check=False)


# %% [code]
# Aggregate per-fold JSONs into one big table via collect_results.py
subprocess.run([
    "python", "collect_results.py",
    "--results-dir", results_dir,
    "--out", os.path.join(results_dir, "summary.json"),
    "--latex", os.path.join(results_dir, "summary.tex"),
], check=False)


# %% [code]
# Quick console preview
with open(os.path.join(results_dir, "summary.json"), "r", encoding="utf-8") as f:
    summary = json.load(f)

print("\n" + "=" * 80)
print(f"{'model':<12} {'dataset':<8} {'frac':<6} {'DSC':<15} {'IoU':<15} {'HD95':<12}")
print("-" * 80)
for row in summary.get("rows", []):
    print(
        f"{row['model']:<12} {row['dataset']:<8} {row['fraction']:<6} "
        f"{row['dice']:<15} {row['iou']:<15} {row['hd95']:<12}"
    )


# %% [code]
# Also print LaTeX
with open(os.path.join(results_dir, "summary.tex"), "r", encoding="utf-8") as f:
    print(f.read())
# >> Output sekmesinden tüm /kaggle/working/results klasörünü indir

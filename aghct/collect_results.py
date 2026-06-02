"""Aggregate per-fold ``evaluate.py`` JSON outputs into a single table.

Reads every ``metrics_*.json`` under ``--results-dir`` and groups them by
``(model, dataset, fraction)``. Outputs:

* ``--out`` (JSON)      — machine-readable summary with mean ± std across folds
* ``--latex`` (.tex)    — paper-ready ``\\begin{table}`` snippet
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
from collections import defaultdict
from statistics import mean, stdev
from typing import Dict, List, Tuple

NAME_RE = re.compile(
    r"^metrics_"
    r"(?P<model>aghct|unet|transunet)_"
    r"(?P<dataset>isic|drive)_"
    r"fold(?P<fold>\d+)"
    r"(?P<rest>.*)\.json$"
)


# ---------------------------------------------------------------------------
def parse_filename(fname: str) -> Dict | None:
    m = NAME_RE.match(fname)
    if not m:
        return None
    return {
        "model": m["model"],
        "dataset": m["dataset"],
        "fold": int(m["fold"]),
        "rest": m["rest"] or "",
    }


def parse_fraction_from_checkpoint(ckpt_path: str) -> float:
    """Extract ``frac<x>`` from the checkpoint path, default 1.0."""
    m = re.search(r"frac([\d.]+)", os.path.basename(ckpt_path))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return 1.0
    return 1.0


# ---------------------------------------------------------------------------
def fmt(mean_v: float, std_v: float, digits: int = 4) -> str:
    return f"{mean_v:.{digits}f} ± {std_v:.{digits}f}"


def aggregate_group(
    entries: List[Dict],
) -> Dict[str, str]:
    """Average per-fold mean metrics into one row."""
    keys = ["dice", "iou", "sensitivity", "specificity", "hd95"]
    row = {}
    for k in keys:
        values = []
        for e in entries:
            v = e["metrics"].get(k, {}).get("mean")
            if v is None:
                continue
            if isinstance(v, float) and (v != v or v == float("inf")):
                # skip NaN / inf when computing the cross-fold mean
                continue
            values.append(v)
        if not values:
            row[k] = "n/a"
        else:
            m = mean(values)
            s = stdev(values) if len(values) > 1 else 0.0
            digits = 2 if k == "hd95" else 4
            row[k] = fmt(m, s, digits)
    row["folds"] = ",".join(str(e["fold"]) for e in entries)
    return row


# ---------------------------------------------------------------------------
def to_latex(rows: List[Dict]) -> str:
    """Build a ``tabular`` summary table."""
    header = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Cross-validated segmentation results (mean $\\pm$ std).}\n"
        "\\label{tab:results}\n"
        "\\begin{tabular}{llcccccc}\n"
        "\\toprule\n"
        "Model & Dataset & Fraction & DSC & IoU & Sens. & Spec. & HD95 \\\\\n"
        "\\midrule\n"
    )
    body = []
    for r in rows:
        body.append(
            f"{r['model'].upper()} & {r['dataset'].upper()} & "
            f"{r['fraction']} & {r['dice']} & {r['iou']} & "
            f"{r['sensitivity']} & {r['specificity']} & {r['hd95']} \\\\"
        )
    footer = "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return header + "\n".join(body).replace("±", "$\\pm$") + footer


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Directory of metrics_*.json files")
    parser.add_argument("--out", default=None, help="Aggregated JSON output path")
    parser.add_argument("--latex", default=None, help="LaTeX output path")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, "metrics_*.json")))
    if not files:
        print(f"[collect_results] no metrics_*.json files in {args.results_dir}")
        return

    grouped: Dict[Tuple[str, str, float], List[Dict]] = defaultdict(list)
    for path in files:
        info = parse_filename(os.path.basename(path))
        if info is None:
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        fraction = parse_fraction_from_checkpoint(data.get("checkpoint", ""))
        key = (info["model"], info["dataset"], fraction)
        grouped[key].append({
            "fold": info["fold"],
            "metrics": data.get("metrics", {}),
            "checkpoint": data.get("checkpoint"),
            "n_params_M": data.get("n_params_M"),
        })

    rows = []
    for (model, dataset, fraction), entries in sorted(grouped.items()):
        row = aggregate_group(entries)
        row.update({
            "model": model,
            "dataset": dataset,
            "fraction": fraction,
            "n_folds": len(entries),
            "n_params_M": entries[0].get("n_params_M"),
        })
        rows.append(row)

    # ---- Console summary ------------------------------------------------
    print("\n" + "=" * 110)
    print(
        f"{'Model':<10} {'Dataset':<8} {'Frac':<6} {'#fold':<6} "
        f"{'DSC':<20} {'IoU':<20} {'HD95':<18}"
    )
    print("-" * 110)
    for r in rows:
        print(
            f"{r['model']:<10} {r['dataset']:<8} {r['fraction']:<6} "
            f"{r['n_folds']:<6} {r['dice']:<20} {r['iou']:<20} {r['hd95']:<18}"
        )
    print("=" * 110)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"rows": rows}, f, indent=2)
        print(f"[collect_results] JSON → {args.out}")

    if args.latex:
        os.makedirs(os.path.dirname(args.latex) or ".", exist_ok=True)
        with open(args.latex, "w", encoding="utf-8") as f:
            f.write(to_latex(rows))
        print(f"[collect_results] LaTeX → {args.latex}")


if __name__ == "__main__":
    main()

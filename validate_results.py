#!/usr/bin/env python3
"""
validate_results.py — concise metrics + plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Checks FPGA predictions against MNIST ground‑truth and, if available,
against a *golden* reference dump. Generates / shows confusion matrix &
ROC plots and prints latency / throughput statistics when present.

Usage
-----
$ python validate_results.py -m metrics/            # default names
$ python validate_results.py -m runs/exp1 \
      --y-hw hw_logits.npy --golden golden.npy \
      --no-show                                    # skip inline display

Outputs
-------
• Accuracy (HW vs ground‑truth)
• Accuracy (HW vs golden) — only if golden file found
• Avg / σ / min / max latency & throughput (optional)
• confusion_matrix.png
• roc_curve.png

All images are saved inside `--metrics-dir` and also displayed unless
`--no-show` is set or the backend is headless.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
)

# --- ensure local project modules are importable ----------
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))  # add <ai_on_fpga/> directory to PYTHONPATH

from mnist_utils import get_mnist_test_labels

# CLI 

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate FPGA predictions vs MNIST ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-m", "--metrics-dir", default="metrics/",
                   help="Directory containing .npy result files")
    p.add_argument("--y-hw", default="y_hw.npy",
                   help="FPGA logits filename inside metrics dir")
    p.add_argument("--golden", default="golden_preds.npy",
                   help="Optional golden logits filename")
    p.add_argument("--latency", default="latency.npy",
                   help="Optional latency file name")
    p.add_argument("--throughput", default="throughput.npy",
                   help="Optional throughput file name")
    p.add_argument("--cm-name", default="confusion_matrix.png",
                   help="Output PNG for confusion matrix")
    p.add_argument("--roc-name", default="roc_curve.png",
                   help="Output PNG for ROC plot")
    p.add_argument("--no-show", action="store_true",
                   help="Do not call plt.show() after saving plots")
    return p.parse_args()

# Helper functions 

def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Return (N,C) float32; convert 1‑D int labels to one‑hot."""
    if arr.ndim == 1:
        C = int(arr.max() + 1)
        onehot = np.zeros((arr.shape[0], C), dtype=np.float32)
        onehot[np.arange(arr.shape[0]), arr.astype(int)] = 1.0
        return onehot
    return arr.astype(np.float32)

def _align(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pad / truncate to same second‑dim."""
    C = max(a.shape[1], b.shape[1])
    def pad(x):
        if x.shape[1] == C:
            return x
        if x.shape[1] < C:
            pad_cols = np.zeros((x.shape[0], C - x.shape[1]), dtype=x.dtype)
            return np.hstack([x, pad_cols])
        return x[:, :C]
    return pad(a), pad(b)

# Plot helpers

def plot_confusion(cm: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_title("Confusion Matrix (HW vs GT)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(cm.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig

def plot_multi_roc(y_true: np.ndarray, y_scores: np.ndarray, out_path: Path):
    n_classes = y_true.shape[1]
    fpr, tpr, roc_auc = {}, {}, {}
    for c in range(n_classes):
        fpr[c], tpr[c], _ = roc_curve(y_true[:, c], y_scores[:, c])
        roc_auc[c] = auc(fpr[c], tpr[c])

    # Macro‑average
    all_fpr = np.unique(np.concatenate([fpr[c] for c in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(all_fpr, mean_tpr, label=f"macro‑avg AUC={macro_auc:.2f}", linewidth=2)
    for c in range(n_classes):
        ax.plot(fpr[c], tpr[c], linestyle="--", label=f"class {c} AUC={roc_auc[c]:.2f}")
    ax.plot([0,1],[0,1], linestyle=":", linewidth=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (One‑vs‑Rest)")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return fig

# Main 

def main():
    args = parse_args()
    mdir = Path(args.metrics_dir)

    # Load required arrays
    try:
        y_hw = np.load(mdir / args.y_hw)
    except FileNotFoundError:
        sys.exit(f"[ERROR] Missing {mdir/args.y_hw}.")

    golden_path = mdir / args.golden
    y_ref = np.load(golden_path) if golden_path.exists() else None

    y_true = get_mnist_test_labels("mnist")

    # Shape prep
    y_true = _ensure_2d(y_true)
    y_hw = _ensure_2d(y_hw)
    y_true, y_hw = _align(y_true, y_hw)
    if y_ref is not None:
        y_ref = _ensure_2d(y_ref)
        y_true, y_ref = _align(y_true, y_ref)

    # Accuracy
    acc_hw = accuracy_score(np.argmax(y_true,1), np.argmax(y_hw,1))
    print(f"HW vs Ground‑Truth Accuracy : {acc_hw*100:.2f}%")
    if y_ref is not None:
        acc_ref = accuracy_score(np.argmax(y_ref,1), np.argmax(y_hw,1))
        print(f"HW vs Golden Accuracy      : {acc_ref*100:.2f}% (golden as truth)")

    # Latency / Throughput
    lat_path = mdir / args.latency
    thr_path = mdir / args.throughput
    if lat_path.exists() and thr_path.exists():
        lat = np.load(lat_path)
        thr = np.load(thr_path)
        print("\nLatency / Throughput:")
        print(f"  latency   : {lat.mean()*1e3:.2f} ms ± {lat.std()*1e3:.2f} (min {lat.min()*1e3:.2f}, max {lat.max()*1e3:.2f})")
        print(f"  throughput: {thr.mean():.1f} inf/s ± {thr.std():.1f}")

    # Confusion Matrix plot
    cm = confusion_matrix(np.argmax(y_true,1), np.argmax(y_hw,1))
    cm_file = mdir / args.cm_name
    fig_cm = plot_confusion(cm, cm_file)
    print(f"Confusion matrix saved → {cm_file}")

    # ROC plot
    roc_file = mdir / args.roc_name
    fig_roc = plot_multi_roc(y_true, y_hw, roc_file)
    print(f"ROC curve saved → {roc_file}")

    if not args.no_show:
        plt.show(block=True)
    else:
        plt.close('all')

if __name__ == "__main__":
    main()

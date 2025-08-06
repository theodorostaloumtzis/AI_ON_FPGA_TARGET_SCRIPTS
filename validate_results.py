#!/usr/bin/env python3
"""
validate_results.py — metrics & plots (no power / idle baseline)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Validates FPGA output (y_hw.npy) against the MNIST ground-truth labels
and, optionally, against a golden reference.  Also prints latency and
throughput statistics gathered during inference.

Typical usage
-------------
$ python validate_results.py -m metrics/
$ python validate_results.py -m metrics/ --no-show            # skip plots
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

from mnist_utils import get_mnist_test_labels               # ground-truth labels

# ------------------------------------------------------------------
# CLI helpers
# ------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-m", "--metrics-dir", default="metrics/", help="Directory containing *.npy results")
    p.add_argument("--y-hw",      default="y_hw.npy",          help="FPGA logits file")
    p.add_argument("--golden",    default="golden_preds.npy",  help="Optional golden logits file")
    # new metric filenames emitted by the revised run-script
    p.add_argument("--latency-comm",   default="latency_comm.npy")
    p.add_argument("--throughput-comm",default="throughput_comm.npy")
    p.add_argument("--latency-inf",    default="latency_inf.npy")
    p.add_argument("--throughput-inf", default="throughput_inf.npy")
    p.add_argument("--cm-name",  default="confusion_matrix.png")
    p.add_argument("--roc-name", default="roc_curve.png")
    p.add_argument("--no-show",  action="store_true", help="Skip plt.show()")
    return p.parse_args()

# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------
def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Convert 1-hot label vector → 2-D logits-like array if needed."""
    if arr.ndim == 1:
        C = int(arr.max() + 1)
        out = np.zeros((arr.size, C), dtype=np.float32)
        out[np.arange(arr.size), arr.astype(int)] = 1.0
        return out
    return arr.astype(np.float32)

def _align(a: np.ndarray, b: np.ndarray):
    """Pad / truncate two logit tensors so they share the same class-dim."""
    C = max(a.shape[1], b.shape[1])
    pad = lambda x: np.hstack([x, np.zeros((x.shape[0], C - x.shape[1]), x.dtype)]) if x.shape[1] < C else x[:, :C]
    return pad(a), pad(b)

def plot_confusion(cm: np.ndarray, path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=.046)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ticks = np.arange(cm.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)

def plot_multi_roc(y_true: np.ndarray, y_scores: np.ndarray, path: Path):
    n_cls = y_true.shape[1]
    fpr, tpr, auc_s = {}, {}, {}
    for c in range(n_cls):
        fpr[c], tpr[c], _ = roc_curve(y_true[:, c], y_scores[:, c])
        auc_s[c] = auc(fpr[c], tpr[c])
    all_fpr = np.unique(np.concatenate([fpr[c] for c in range(n_cls)]))
    mean_tpr = np.mean([np.interp(all_fpr, fpr[c], tpr[c]) for c in range(n_cls)], axis=0)
    macro_auc = auc(all_fpr, mean_tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(all_fpr, mean_tpr, label=f"macro AUC = {macro_auc:.2f}", lw=2)
    for c in range(n_cls):
        ax.plot(fpr[c], tpr[c], "--", label=f"class {c} AUC = {auc_s[c]:.2f}")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    mdir = Path(args.metrics_dir)

    # --- predictions ---------------------------------------------------
    y_hw   = _ensure_2d(np.load(mdir / args.y_hw))
    y_true = _ensure_2d(get_mnist_test_labels("mnist"))  # 1-hot labels
    y_true, y_hw = _align(y_true, y_hw)

    if (mdir / args.golden).exists():
        y_ref = _ensure_2d(np.load(mdir / args.golden))
        y_true, y_ref = _align(y_true, y_ref)
    else:
        y_ref = None

    print(f"HW vs GT Accuracy    : {accuracy_score(y_true.argmax(1), y_hw.argmax(1)) * 100:.2f}%")
    if y_ref is not None:
        print(f"HW vs Golden Accuracy: {accuracy_score(y_ref.argmax(1), y_hw.argmax(1)) * 100:.2f}%")

    # --- latency & throughput -----------------------------------------
    metric_pairs = [
        ("COMMS",    mdir / args.latency_comm,    mdir / args.throughput_comm),
        ("INFERENCE",mdir / args.latency_inf,     mdir / args.throughput_inf),
    ]
    for tag, lat_p, thr_p in metric_pairs:
        if lat_p.exists() and thr_p.exists():
            lat, thr = np.load(lat_p), np.load(thr_p)
            print(f"\n[{tag}]")
            print(f"  Latency   : {lat.mean()*1e3:.4f} ms ± {lat.std()*1e3:.4f}  "
                  f"(min={lat.min()*1e3:.4f}, max={lat.max()*1e3:.4f})")
            print(f"  Throughput: {thr.mean():.2f} inf/s ± {thr.std():.2f}")

    # --- plots ---------------------------------------------------------
    plot_confusion(confusion_matrix(y_true.argmax(1), y_hw.argmax(1)), mdir / args.cm_name)
    plot_multi_roc(y_true, y_hw, mdir / args.roc_name)
    print(f"Confusion matrix saved → {mdir / args.cm_name}")
    print(f"ROC curve saved        → {mdir / args.roc_name}")

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")

if __name__ == "__main__":
    main()

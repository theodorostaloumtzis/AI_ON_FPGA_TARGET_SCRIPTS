#!/usr/bin/env python3
"""
validate_results.py — metrics & plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Validates FPGA output (y_hw.npy) against the MNIST ground-truth labels and
(optional) golden reference; prints robust latency / throughput statistics
(with P50/P90/P99 and optional outlier trimming) and raw cycle counts.
Saves a confusion-matrix PNG and a multi-class ROC curve PNG.

Examples
--------
python validate_results.py -m metrics/
python validate_results.py -m metrics/ --clk-mhz 150 --exclude-pct 1
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

from mnist_utils import get_mnist_test_labels  # ground-truth labels
# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("-m", "--metrics-dir", default="metrics/",
                   help="Directory containing *.npy results")
    p.add_argument("--y-hw",      default="y_hw.npy",
                   help="FPGA logits file")
    p.add_argument("--golden",    default="golden_preds.npy",
                   help="Optional golden-reference logits file")
    # timing / throughput arrays
    p.add_argument("--latency-comm",    default="latency_comm.npy")
    p.add_argument("--throughput-comm", default="throughput_comm.npy")
    p.add_argument("--latency-inf",     default="latency_inf.npy")
    p.add_argument("--throughput-inf",  default="throughput_inf.npy")
    # raw cycles
    p.add_argument("--cycles",    default="cycles_raw.npy",
                   help="Optional raw cycle-count file")
    p.add_argument("--clk-mhz",   type=float, default=None,
                   help="FPGA clock in MHz for cycles→µs conversion")
    # outlier trimming & plotting
    p.add_argument("--exclude-pct", type=float, default=0.0,
                   help="Trim the slowest P%% latency samples (0 = keep all)")
    p.add_argument("--cm-name",  default="confusion_matrix.png")
    p.add_argument("--roc-name", default="roc_curve.png")
    p.add_argument("--no-show",  action="store_true",
                   help="Skip plt.show() (useful in headless runs)")
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure logits-like 2-D array (N,C). Converts 1-D label vector to one-hot."""
    if arr.ndim == 1:
        C = int(arr.max() + 1)
        out = np.zeros((arr.size, C), dtype=np.float32)
        out[np.arange(arr.size), arr.astype(int)] = 1.0
        return out
    return arr.astype(np.float32)

def _align(a: np.ndarray, b: np.ndarray):
    """Pad / truncate so both arrays share the same class dimension."""
    C = max(a.shape[1], b.shape[1])
    def pad(x):
        if x.shape[1] < C:
            pad_cols = np.zeros((x.shape[0], C - x.shape[1]), dtype=x.dtype)
            return np.hstack([x, pad_cols])
        return x[:, :C]
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
    """Macro-averaged ROC and per-class curves."""
    n_cls = y_true.shape[1]
    fpr, tpr, auc_s = {}, {}, {}
    for c in range(n_cls):
        fpr[c], tpr[c], _ = roc_curve(y_true[:, c], y_scores[:, c])
        auc_s[c] = auc(fpr[c], tpr[c])
    all_fpr = np.unique(np.concatenate([fpr[c] for c in range(n_cls)]))
    mean_tpr = np.mean(
        [np.interp(all_fpr, fpr[c], tpr[c]) for c in range(n_cls)], axis=0
    )
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

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    mdir = Path(args.metrics_dir)

    # ── predictions ────────────────────────────────────────────────────
    y_hw   = _ensure_2d(np.load(mdir / args.y_hw))
    y_true = _ensure_2d(get_mnist_test_labels("mnist"))
    y_true, y_hw = _align(y_true, y_hw)

    y_ref = None
    if (mdir / args.golden).exists():
        y_ref = _ensure_2d(np.load(mdir / args.golden))
        y_true, y_ref = _align(y_true, y_ref)

    print(f"HW vs GT Accuracy    : "
          f"{accuracy_score(y_true.argmax(1), y_hw.argmax(1)) * 100:.2f}%")
    if y_ref is not None:
        print(f"HW vs Golden Accuracy: "
              f"{accuracy_score(y_ref.argmax(1), y_hw.argmax(1)) * 100:.2f}%")

    # ── load timing arrays ─────────────────────────────────────────────
    paths = {
        "COMMS_lat": mdir / args.latency_comm,
        "COMMS_thr": mdir / args.throughput_comm,
        "INF_lat":   mdir / args.latency_inf,
        "INF_thr":   mdir / args.throughput_inf,
    }
    arrays = {k: (np.load(p) if p.exists() else None) for k, p in paths.items()}

    # ── outlier trimming & masks ───────────────────────────────────────
    mask_inf = None
    if arrays["INF_lat"] is not None and args.exclude_pct > 0.0:
        cut = 100.0 - args.exclude_pct
        thr = np.percentile(arrays["INF_lat"], cut)
        mask_inf = arrays["INF_lat"] <= thr
        arrays["INF_lat"] = arrays["INF_lat"][mask_inf]
        arrays["INF_thr"] = arrays["INF_thr"][mask_inf]

    if arrays["COMMS_lat"] is not None and args.exclude_pct > 0.0:
        cut = 100.0 - args.exclude_pct
        thr = np.percentile(arrays["COMMS_lat"], cut)
        m = arrays["COMMS_lat"] <= thr
        arrays["COMMS_lat"], arrays["COMMS_thr"] = arrays["COMMS_lat"][m], arrays["COMMS_thr"][m]

    # ── print stats ────────────────────────────────────────────────────
    for tag in ("COMMS", "INF"):
        lat, thr = arrays[f"{tag}_lat"], arrays[f"{tag}_thr"]
        if lat is None or thr is None:
            continue
        lat_ms = lat * 1e3
        p50, p90, p99 = np.percentile(lat_ms, [50, 90, 99])
        print(f"\n[{tag}]")
        print(f"  Latency   : {lat_ms.mean():.4f} ms ± {lat_ms.std():.4f}  "
              f"(min={lat_ms.min():.4f}, max={lat_ms.max():.4f})")
        print(f"            : P50={p50:.4f}  P90={p90:.4f}  P99={p99:.4f}")
        print(f"  Throughput: {thr.mean():.2f} inf/s ± {thr.std():.2f}")

    # ── cycle counts ──────────────────────────────────────────────────
    cyc_path = mdir / args.cycles
    if cyc_path.exists():
        cycles = np.load(cyc_path)
        if mask_inf is not None:
            cycles = cycles[mask_inf]          # align with INF mask
        p50c, p90c, p99c = np.percentile(cycles, [50, 90, 99])
        print(f"\n[CYCLES]   ({cycles.size} samples)")
        print(f"  Mean cycles : {cycles.mean():.0f} ± {cycles.std():.0f}  "
              f"(min={cycles.min():.0f}, max={cycles.max():.0f})")
        print(f"            : P50={p50c:.0f}  P90={p90c:.0f}  P99={p99c:.0f}")
        if args.clk_mhz:
            clk_hz = args.clk_mhz * 1e6
            lat_us = cycles / clk_hz * 1e6
            print(f"  Mean time  : {lat_us.mean():.3f} µs "
                  f"({1/lat_us.mean()*1e6:.2f} inf/s @ {args.clk_mhz:.0f} MHz)")

    # ── plots ─────────────────────────────────────────────────────────
    plot_confusion(confusion_matrix(y_true.argmax(1), y_hw.argmax(1)),
                   mdir / args.cm_name)
    plot_multi_roc(y_true, y_hw, mdir / args.roc_name)
    print(f"Confusion matrix saved → {mdir / args.cm_name}")
    print(f"ROC curve saved        → {mdir / args.roc_name}")

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")

if __name__ == "__main__":
    main()

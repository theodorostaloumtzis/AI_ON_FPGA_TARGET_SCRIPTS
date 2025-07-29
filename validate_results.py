#!/usr/bin/env python3
"""
validate_results.py — complete metrics + plots (idle-aware)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Validates FPGA output (from y_hw.npy) against MNIST labels and optionally against a golden reference.
Adds latency/throughput and detailed **energy** stats that subtract an idle-baseline power.

Usage examples
--------------
$ python validate_results.py -m metrics/rf_analysis \
      --idle-power idle_power.npy
$ python validate_results.py -m metrics/ --no-show         # skip plots on headless box
"""
from __future__ import annotations
import argparse, warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from mnist_utils import get_mnist_test_labels

# ------------------------------------------------------------------
# CLI helpers
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-m", "--metrics-dir", default="metrics/", help="Directory with *.npy results")
    p.add_argument("--y-hw",          default="y_hw.npy",          help="FPGA logits file")
    p.add_argument("--golden",        default="golden_preds.npy",  help="Optional golden logits")
    p.add_argument("--idle-power",    default="idle_power.npy",                 help="npy file with idle mW samples")
    p.add_argument("--cm-name",       default="confusion_matrix.png")
    p.add_argument("--roc-name",      default="roc_curve.png")
    p.add_argument("--no-show", action="store_true", help="Skip plt.show()")
    return p.parse_args()

# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        C = int(arr.max() + 1)
        out = np.zeros((arr.size, C), dtype=np.float32)
        out[np.arange(arr.size), arr.astype(int)] = 1.0
        return out
    return arr.astype(np.float32)

def _align(a: np.ndarray, b: np.ndarray):
    C = max(a.shape[1], b.shape[1])
    pad = lambda x: np.hstack([x, np.zeros((x.shape[0], C - x.shape[1]), x.dtype)]) if x.shape[1] < C else x[:, :C]
    return pad(a), pad(b)

def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=.046)
    ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ticks = np.arange(cm.shape[0]); ax.set_xticks(ticks); ax.set_yticks(ticks)
    thresh = cm.max()/2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,cm[i,j],ha="center",va="center",color="white" if cm[i,j]>thresh else "black")
    fig.tight_layout(); fig.savefig(path,dpi=150)

def plot_multi_roc(y_true,y_scores,path):
    n_cls = y_true.shape[1]
    fpr,tpr,auc_s = {},{},{ }
    for c in range(n_cls):
        fpr[c], tpr[c], _ = roc_curve(y_true[:,c], y_scores[:,c])
        auc_s[c] = auc(fpr[c], tpr[c])
    all_fpr = np.unique(np.concatenate([fpr[c] for c in range(n_cls)]))
    mean_tpr = np.mean([np.interp(all_fpr,fpr[c],tpr[c]) for c in range(n_cls)], axis=0)
    macro_auc = auc(all_fpr, mean_tpr)
    fig,ax = plt.subplots(figsize=(6,5))
    ax.plot(all_fpr, mean_tpr,label=f"macro AUC={macro_auc:.2f}",lw=2)
    for c in range(n_cls):
        ax.plot(fpr[c],tpr[c],'--',label=f"class {c} AUC={auc_s[c]:.2f}")
    ax.plot([0,1],[0,1],'k:',lw=1); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize="small"); fig.tight_layout(); fig.savefig(path,dpi=150)

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()
    mdir = Path(args.metrics_dir)

    # --- predictions ---
    y_hw   = _ensure_2d(np.load(mdir/args.y_hw))
    y_true = _ensure_2d(get_mnist_test_labels("mnist"))
    y_true,y_hw = _align(y_true,y_hw)

    if (mdir/args.golden).exists():
        y_ref = _ensure_2d(np.load(mdir/args.golden))
        y_true, y_ref = _align(y_true,y_ref)
    else:
        y_ref = None

    print(f"HW vs GT Accuracy   : {accuracy_score(y_true.argmax(1), y_hw.argmax(1))*100:.2f}%")
    if y_ref is not None:
        print(f"HW vs Golden Accuracy: {accuracy_score(y_ref.argmax(1), y_hw.argmax(1))*100:.2f}%")

    # --- latency / throughput ---
    for tag in ("1","2"):
        lat_p, thr_p = mdir/f"latency{tag}.npy", mdir/f"throughput{tag}.npy"
        if lat_p.exists() and thr_p.exists():
            lat, thr = np.load(lat_p), np.load(thr_p)
            print(f"\n[Latency/Throughput {tag}]\n  Latency   : {lat.mean()*1e3:.4f} ms ± {lat.std()*1e3:.4f}  (min={lat.min()*1e3:.4f}, max={lat.max()*1e3:.4f})")
            print(f"  Throughput: {thr.mean():.2f} inf/s ± {thr.std():.2f}")

    # ------------------------------------------------------------------
    # Power / Energy
    # ------------------------------------------------------------------
    tr_path, bd_path = mdir/"power_trace.npy", mdir/"power_bounds.npy"
    if tr_path.exists() and bd_path.exists():
        trace  = np.load(tr_path, allow_pickle=True).tolist()
        bounds = np.load(bd_path, allow_pickle=True).tolist()
        if not trace or not bounds:
            print("\n[Power] Empty trace or bounds – skipping energy calc.")
        else:
            ts = np.array([t for t,_ in trace])
            pw = np.array([p for _,p in trace])  # mW

            # --- idle baseline ---
            if args.idle_power and Path(args.idle_power).exists():
                idle_samples = np.load(args.idle_power).astype(float)
                P_idle = idle_samples.mean()
                print(f"[Power] Using idle baseline file ⇒ {P_idle:.2f} mW")
            else:
                win = min( max(len(ts)//20, 50), len(ts) )  # first ~5% or at least 50 pts
                P_idle = pw[:win].mean()
                print(f"[Power] Estimated idle baseline ⇒ {P_idle:.2f} mW (first {win} pts)")

            pw_net = np.clip(pw - P_idle, 0, None)
            E_total_mJ = np.trapz(pw_net, ts)

            # per-sample energy
            e_per = []
            for t0,t1 in bounds:
                idx = np.where((ts>=t0)&(ts<=t1))[0]
                if idx.size>=2:
                    e_per.append( np.trapz(pw_net[idx], ts[idx]) )
            avg_e = np.mean(e_per) if e_per else 0.0

            print("\n[Power Summary]")
            print(f"  Avg Net Power: {pw_net.mean():.2f} mW  (σ={pw_net.std():.2f})")
            print(f"  Min / Max Net: {pw_net.min():.2f} / {pw_net.max():.2f} mW")
            print(f"  Total Energy (net): {E_total_mJ:.2f} mJ")
            print(f"  Idle Power Avg: {P_idle:.2f} mW")
            if e_per:
                print(f"  Avg Energy/sample: {avg_e:.4f} mJ (N={len(e_per)})")
            else:
                print("  Per-sample energy unavailable (need ≥2 INA points per bound)")

            # histogram
            # --- histogram (skip zeros) ---------------------------------
            non_zero = pw_net[pw_net > 1e-3]      # drop clipped-to-zero points
            if non_zero.size == 0:
                print("  (all net power samples ≈ 0 mW after idle subtraction; histogram skipped)")
            else:
                fig, ax = plt.subplots()
                ax.hist(non_zero, bins=40, color="steelblue", edgecolor="black")
                ax.set_title("Net Power Histogram (INA260 − idle)")
                ax.set_xlabel("Power (mW)")
                ax.set_ylabel("Frequency")
                fig.tight_layout()
                hist_path = mdir / "power_histogram.png"
                fig.savefig(hist_path, dpi=150)
                print(f"  Power histogram saved → {hist_path}")

    else:
        print("\n[Power] Trace/bounds not found – skipping power analysis.")

    # ------------------------------------------------------------------
    # Confusion + ROC
    # ------------------------------------------------------------------
    plot_confusion(confusion_matrix(y_true.argmax(1), y_hw.argmax(1)), mdir/args.cm_name)
    plot_multi_roc(y_true, y_hw, mdir/args.roc_name)
    print(f"Confusion matrix saved → {mdir/args.cm_name}")
    print(f"ROC curve saved        → {mdir/args.roc_name}")

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")

if __name__ == "__main__":
    main()
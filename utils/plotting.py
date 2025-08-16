from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_confusion(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=.046)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ticks = np.arange(cm.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    thresh = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_multi_roc(y_true: np.ndarray, y_scores: np.ndarray, path: Path) -> None:
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
    plt.close(fig)

def plot_power_trace(p_abs: np.ndarray | None,
                     p_dyn: np.ndarray | None,
                     out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if p_abs is not None and p_abs.size:
        ax.plot(p_abs[:, 0] - p_abs[0, 0], p_abs[:, 1], label="Absolute (mW)")
    if p_dyn is not None and p_dyn.size:
        ax.plot(p_dyn[:, 0] - p_dyn[0, 0], p_dyn[:, 1], label="Dynamic (mW)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (mW)")
    ax.set_title("Power Trace")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

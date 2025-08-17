# comp_utils/metrics_plot.py
from __future__ import annotations
from typing import Dict, Any, Sequence, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _safe_stats(m: Dict[str, Any], key: str) -> Dict[str, float]:
    x = m.get(key)
    return x if isinstance(x, dict) else {}


def _save(fig: plt.Figure, save: Optional[str | Path]) -> None:
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")


def plot_eval_summary(metrics: Dict[str, Any],
                      *,
                      title: Optional[str] = None,
                      save: Optional[str | Path] = None) -> plt.Figure:
    """
    One-run summary figure:
      • Latency per sample (mean ± std) with P50/P90/P99 annotations
      • Throughput (batchwise mean ± std) + global line
      • Info panel (backend, accuracy, samples, batches)
    """
    lat = _safe_stats(metrics, "latency_ms_per_sample")
    thr = _safe_stats(metrics, "throughput_ips_batchwise")
    thr_global = metrics.get("throughput_ips_global", None)
    backend = metrics.get("backend", "?")
    acc = float(metrics.get("accuracy", 0.0)) * 100.0
    n_samples = int(metrics.get("n_samples", 0))
    n_batches = int(metrics.get("n_batches", 0))

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))

    # Latency panel
    ax = axs[0]
    if lat:
        mean, std = lat["mean"], lat["std"]
        ax.bar([0], [mean], width=0.6)
        ax.errorbar([0], [mean], yerr=[std], fmt="none", capsize=6)
        ax.set_xticks([0])
        ax.set_xticklabels(["Latency\n(ms/sample)"])
        ax.set_ylabel("ms / sample")
        ymax = ax.get_ylim()[1]
        ax.text(0, ymax * 0.95,
                f"P50={lat['p50']:.3f}\nP90={lat['p90']:.3f}\nP99={lat['p99']:.3f}",
                ha="center", va="top", fontsize=9)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No latency stats", ha="center", va="center")

    # Throughput panel
    ax = axs[1]
    if thr:
        mean, std = thr["mean"], thr["std"]
        ax.bar([0], [mean], width=0.6, label="Batchwise mean")
        ax.errorbar([0], [mean], yerr=[std], fmt="none", capsize=6)
        if thr_global is not None and np.isfinite(thr_global):
            ax.axhline(thr_global, ls="--", lw=1.5, label=f"Global {thr_global:.1f} ips")
        ax.set_xticks([0])
        ax.set_xticklabels(["Throughput\n(inf/s)"])
        ax.set_ylabel("inferences / second")
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No throughput stats", ha="center", va="center")

    # Info panel
    ax = axs[2]
    ax.axis("off")
    ax.text(0.0, 1.0,
            f"Backend : {backend}\n"
            f"Accuracy: {acc:.2f}%\n"
            f"Samples : {n_samples}\n"
            f"Batches : {n_batches}",
            va="top", family="monospace", fontsize=11)

    if title:
        fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout()
    _save(fig, save)
    return fig


def plot_compare_latency(results: Sequence[Dict[str, Any]],
                         labels: Sequence[str],
                         *,
                         title: str = "Latency per sample (mean ± std)",
                         save: Optional[str | Path] = None) -> plt.Figure:
    """Grouped bars comparing latency across runs."""
    assert len(results) == len(labels), "results/labels length mismatch"
    means = [ _safe_stats(m, "latency_ms_per_sample").get("mean", np.nan) for m in results ]
    stds  = [ _safe_stats(m, "latency_ms_per_sample").get("std", 0.0) for m in results ]

    x = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(max(6.0, 1.6 * len(results)), 4.2))
    ax.bar(x, means, yerr=stds, capsize=6, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("ms / sample")
    ax.set_title(title)

    # annotate a couple of percentiles if present
    ymax = np.nanmax(np.array(means) + np.array(stds)) if len(means) else 1.0
    ymax = float(ymax) if np.isfinite(ymax) else 1.0
    for i, m in enumerate(results):
        lat = _safe_stats(m, "latency_ms_per_sample")
        if lat:
            ax.text(i, (means[i] if np.isfinite(means[i]) else 0) + 0.02 * ymax,
                    f"P50={lat['p50']:.2f}\nP90={lat['p90']:.2f}",
                    ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    _save(fig, save)
    return fig


def plot_compare_throughput(results: Sequence[Dict[str, Any]],
                            labels: Sequence[Dict[str, Any]],
                            *,
                            title: str = "Throughput (batchwise mean ± std, global line)",
                            save: Optional[str | Path] = None) -> plt.Figure:
    """Grouped bars comparing throughput across runs, with global markers."""
    assert len(results) == len(labels), "results/labels length mismatch"
    means   = [ _safe_stats(m, "throughput_ips_batchwise").get("mean", np.nan) for m in results ]
    stds    = [ _safe_stats(m, "throughput_ips_batchwise").get("std", 0.0) for m in results ]
    globals_ = [ m.get("throughput_ips_global", np.nan) for m in results ]

    x = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(max(6.0, 1.6 * len(results)), 4.2))
    ax.bar(x, means, yerr=stds, capsize=6, width=0.6, label="Batchwise mean")
    ax.scatter(x, globals_, marker="D", s=36, zorder=3, label="Global")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("inferences / second")
    ax.set_title(title)
    ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, save)
    return fig

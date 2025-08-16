from __future__ import annotations
from pathlib import Path
import numpy as np

def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """
    Make logits/probs 2D; if 1D labels are passed, one-hot them.
    """
    if arr.ndim == 1:
        C = int(arr.max() + 1)
        out = np.zeros((arr.size, C), dtype=np.float32)
        out[np.arange(arr.size), arr.astype(int)] = 1.0
        return out
    return arr.astype(np.float32)

def align_cols(a: np.ndarray, b: np.ndarray):
    """
    Column-align two [N×C] arrays by padding/truncating to the same C.
    """
    C = max(a.shape[1], b.shape[1])

    def pad(x: np.ndarray) -> np.ndarray:
        if x.shape[1] < C:
            pad_cols = np.zeros((x.shape[0], C - x.shape[1]), dtype=x.dtype)
            return np.hstack([x, pad_cols])
        return x[:, :C]

    return pad(a), pad(b)

def safe_load(path: Path) -> np.ndarray | None:
    return np.load(path) if Path(path).exists() else None

def trim_outliers(latency_s: np.ndarray, exclude_pct: float):
    """
    Trim the slowest `exclude_pct` percent of samples from a 1D latency array (seconds).
    Returns (trimmed_array, mask).
    """
    if latency_s.size == 0 or exclude_pct <= 0.0:
        return latency_s, None
    cut = 100.0 - float(exclude_pct)
    thr = np.percentile(latency_s, cut)
    mask = latency_s <= thr
    return latency_s[mask], mask

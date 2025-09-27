#!/usr/bin/env python3
"""
svhn_utils.py — shared utilities for SVHN Q6.10 fixed-point workflows
---------------------------------------------------------------------
* Downloads SVHN `.mat` files (train/test) from multiple mirrors and caches
  them under `svhn/` (no TF required).
* Loads with scipy.io.loadmat (MAT v5/7), reshapes to NHWC, normalises to
  [0,1], and quantises to Q6.10 (ap_fixed<16,6>) as int16.
* Exposes a small API compatible with your MNIST helper.
"""
from __future__ import annotations
import os, ssl, shutil, urllib.request
from typing import Tuple, Literal

import numpy as np

# Load MATLAB v5/7 .mat files (SVHN official format)
try:
    from scipy.io import loadmat
except Exception as e:
    raise ImportError(
        "svhn_utils.py requires 'scipy' to read SVHN .mat files.\n"
        "Install it with: pip install scipy"
    ) from e

# Allow HTTPS even when CA bundle is missing (common on embedded distros)
ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore

# Fixed-point parameters for ap_fixed<16,6> (total=16, integer=6, frac=10)
FRAC_BITS: int = 10
SCALE: int = 1 << FRAC_BITS  # 1024

def encode_arr(x_f32: np.ndarray) -> np.ndarray:
    """Vectorised encoder: float32 → int16 (Q6.10)."""
    return np.round(x_f32 * SCALE).astype(np.int16, copy=False)

def decode_arr(x_i16: np.ndarray) -> np.ndarray:
    """Vectorised decoder: int16 (Q6.10) → float32."""
    return x_i16.astype(np.float32, copy=False) / SCALE


# ---------------------------------------------------------------------
# Robust SVHN loader (.mat, MATLAB v5/7)
# Files are shaped as:
#   X: (32, 32, 3, N), uint8   images
#   y: (N, 1) or (N,), uint8 with '10' meaning digit 0
# We return:
#   images_f32: (N, 32, 32, 3), float32 in [0,1]
#   labels: (N,), uint8 in 0..9
# ---------------------------------------------------------------------
_BASE_NAMES = {
    "train": "train_32x32.mat",
    "test":  "test_32x32.mat",
    # (extra set can be added the same way if you need it)
}

MIRRORS = [
    "http://ufldl.stanford.edu/housenumbers/",
    "http://benchmark.ini.rub.de/Dataset/SVHN/cropped/",
    "https://benchmark.ini.rub.de/Dataset/SVHN/cropped/",
]

def _download_with_mirrors(basename: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for base in MIRRORS:
        url = base + basename
        try:
            print(f"[svhn_utils] Downloading {url}")
            with urllib.request.urlopen(url, timeout=30) as r, open(dst, "wb") as f:
                shutil.copyfileobj(r, f)
            print(f"[svhn_utils] Saved → {dst}")
            return
        except Exception as e:
            print(f"[svhn_utils] WARN {e.__class__.__name__}: {e}")
    raise RuntimeError(f"All mirrors failed for {basename}")

def _ensure_local_file(cache_dir: str, split: Literal["train","test"]) -> str:
    fname = _BASE_NAMES[split]
    local = os.path.join(cache_dir, fname)
    if not os.path.exists(local):
        _download_with_mirrors(fname, local)
    return local

def _load_svhn_mat(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read SVHN .mat into float32 images [0,1], uint8 labels 0..9."""
    m = loadmat(path)
    if "X" not in m or "y" not in m:
        raise KeyError(f"{path} does not contain 'X'/'y' arrays")

    X = m["X"]  # (32,32,3,N), uint8
    y = m["y"]  # (N,1) or (N,), uint8 with 10 as '0'

    # Move axis to (N, 32, 32, 3)
    X = np.transpose(X, (3, 0, 1, 2)).astype(np.float32) / 255.0
    y = np.array(y).reshape(-1).astype(np.uint8)
    y = (y % 10).astype(np.uint8)  # map 10→0
    return X, y

# ---------------------------------------------------------------------
# Public API (mirrors your MNIST helper)
# ---------------------------------------------------------------------
def load_and_quantize_svhn(
    cache_dir: str = "svhn",
    split: Literal["test", "train"] = "test",
    flatten: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Download (if needed), normalise, and quantise SVHN split.

    Parameters
    ----------
    cache_dir : str
        Where to cache the .mat file(s).
    split : {'test','train'}
        Which split to load.
    flatten : bool
        If True, return X as (N, 3072) int16; else (N, 32, 32, 3) int16.

    Returns
    -------
    X_i16 : np.ndarray
        Q6.10-quantised images, dtype=int16. Shape (N, 3072) if `flatten`
        else (N, 32, 32, 3).
    y_uint8 : np.ndarray
        Integer class labels 0–9, shape (N,), dtype=uint8.
    """
    os.makedirs(cache_dir, exist_ok=True)
    local = _ensure_local_file(cache_dir, split)
    images_f32, labels = _load_svhn_mat(local)
    if flatten:
        images_f32 = images_f32.reshape(len(images_f32), -1)  # (N, 3072)
    X_i16 = encode_arr(images_f32)
    return X_i16, labels

def get_svhn_test_labels(cache_dir: str = "svhn") -> np.ndarray:
    """Return only the SVHN test labels (0–9) without image data."""
    _, labels = load_and_quantize_svhn(cache_dir=cache_dir, split="test")
    return labels

# ---------------------------------------------------------------------
# Optional: tiny CLI for quick checks
# ---------------------------------------------------------------------
if __name__ == "__main__":
    X_i16, y = load_and_quantize_svhn()
    print("SVHN test:", X_i16.shape, X_i16.dtype, y.shape, y.dtype)
    print("First 10 labels:", y[:10].tolist())

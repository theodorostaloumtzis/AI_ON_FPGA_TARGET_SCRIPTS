#!/usr/bin/env python3
"""
mnist_utils.py — shared utilities for MNIST Q6.10 fixed-point workflows
----------------------------------------------------------------------
* Downloads the MNIST **test** IDX files from multiple mirrors and caches them
  under `mnist/` (HTTPS fallback, tiny footprint, no TensorFlow needed).
* Provides helpers to normalise, quantise (`ap_fixed<16,6>`), and decode.
"""
from __future__ import annotations
import numpy as np
import os, urllib.request, gzip, struct, shutil, ssl
from typing import Tuple

# Allow HTTPS even when CA bundle is missing (common on embedded distros)
ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore

# -----------------------------------------------------------------------------
# Fixed-point parameters for ap_fixed<16,6> (total=16, integer=6, frac=10)
# -----------------------------------------------------------------------------
FRAC_BITS: int = 10
SCALE: int = 1 << FRAC_BITS  # 2¹⁰ = 1024

def encode_arr(x_f32: np.ndarray) -> np.ndarray:
    """Vectorised encoder: float32 → int16 (Q6.10)."""
    return np.round(x_f32 * SCALE).astype(np.int16, copy=False)

def decode_arr(x_i16: np.ndarray) -> np.ndarray:
    """Vectorised decoder: int16 (Q6.10) → float32."""
    return x_i16.astype(np.float32, copy=False) / SCALE

# -----------------------------------------------------------------------------
# Robust MNIST loader (IDX format) — tries multiple mirrors
# -----------------------------------------------------------------------------
_BASE_NAMES = {
    "images": "t10k-images-idx3-ubyte.gz",
    "labels": "t10k-labels-idx1-ubyte.gz",
}
MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",   # Google mirror
    "http://yann.lecun.com/exdb/mnist/",                     # Original site
]

def _download_with_mirrors(basename: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for base in MIRRORS:
        url = base + basename
        try:
            print(f"[mnist_utils] Downloading {url}")
            with urllib.request.urlopen(url, timeout=20) as r, open(dst, "wb") as f:
                shutil.copyfileobj(r, f)
            print(f"[mnist_utils] Saved → {dst}")
            return
        except Exception as e:
            print(f"[mnist_utils] WARN {e.__class__.__name__}: {e}")
    raise RuntimeError(f"All mirrors failed for {basename}")

def _load_idx(img_path: str, lbl_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with gzip.open(img_path, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    with gzip.open(lbl_path, "rb") as f:
        _, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return images, labels

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def load_and_quantize_mnist(cache_dir: str = "mnist") -> Tuple[np.ndarray, np.ndarray]:
    """Download (if needed), normalise, and quantise MNIST test set.

    Returns
    -------
    X_test_i16 : np.ndarray
        Shape (N, 784), dtype int16 — each pixel in Q6.10 format.
    y_test_int : np.ndarray
        Shape (N,), dtype uint8 — integer class labels 0-9.
    """
    os.makedirs(cache_dir, exist_ok=True)
    paths = {}
    for key, fname in _BASE_NAMES.items():
        local = os.path.join(cache_dir, fname)
        if not os.path.exists(local):
            _download_with_mirrors(fname, local)
        paths[key] = local

    images_u8, labels = _load_idx(paths["images"], paths["labels"])
    images_f32 = images_u8.astype(np.float32) / 255.0        # 0-1
    X_test_i16 = encode_arr(images_f32.reshape(len(images_f32), -1))
    return X_test_i16, labels


def get_mnist_test_labels(cache_dir: str = "mnist") -> np.ndarray:
    """Return only the MNIST test labels (0–9) without image data."""
    _, labels = load_and_quantize_mnist(cache_dir)
    return labels


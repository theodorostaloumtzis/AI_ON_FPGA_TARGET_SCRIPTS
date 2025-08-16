from __future__ import annotations
import numpy as np
from tqdm.auto import tqdm

def pack4(X_i16: np.ndarray, batch_size: int = 1_000) -> np.ndarray:
    """
    Pack 4×int16 pixels → 1×uint64 word per group of four features.
    """
    N, F = X_i16.shape
    assert F % 4 == 0, "Feature dimension must be divisible by 4"
    X_u64 = np.empty((N, F // 4), dtype=np.uint64)
    shifts = np.array([0, 16, 32, 48], dtype=np.uint64).reshape(1, 1, 4)
    for i in tqdm(range(0, N, batch_size), desc="Packing MNIST", unit="samples"):
        chunk  = X_i16[i:i + batch_size].astype(np.uint64)
        chunk  = chunk.reshape(-1, F // 4, 4)
        packed = ((chunk & 0xFFFF) << shifts).sum(axis=2, dtype=np.uint64)
        X_u64[i:i + batch_size] = packed
    return X_u64

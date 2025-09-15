# comp_utils/power_csv_simple.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def load_hw_csv_simple(path: str | Path) -> pd.DataFrame:
    """
    Load your hardware CSV, strip header whitespace, and create:
      - df['timestamp'] (datetime)
      - df['t_s']       (seconds since start)
    Works with GPU-Z (Date+Time), uProf/PowerGadget (TIME STAMP or Elapsed Time).
    """
    df = pd.read_csv(path)
    # strip weird leading/trailing spaces from headers (your file has them)
    df.columns = df.columns.str.strip()

    # Build a timestamp
    ts = None
    if {"Date", "Time"}.issubset(df.columns):
        ts = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce")
    elif "TIME STAMP" in df.columns:
        ts = pd.to_datetime(df["TIME STAMP"], errors="coerce")
    elif "Elapsed Time (sec)" in df.columns:
        # numeric seconds → fabricate a datetime axis
        secs = pd.to_numeric(df["Elapsed Time (sec)"], errors="coerce")
        ts = pd.to_datetime(secs - secs.min(), unit="s", origin="unix")
    else:
        # last resort: try to parse the first column
        ts = pd.to_datetime(df.iloc[:, 0], errors="coerce")

    # keep only rows where we could parse time
    df = df[ts.notna()].copy()
    df["timestamp"] = ts[ts.notna()]
    df["t_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    return df

def list_columns(df: pd.DataFrame) -> list[str]:
    """See exact column names after header cleanup."""
    return list(df.columns)

def plot_series(df: pd.DataFrame, y_col: str, *, time_col: str = "timestamp",
                title: str | None = None, ylabel: str | None = None) -> None:
    """
    Minimal plotting helper: plot any column vs time.
    Example y_col: 'GPU 1 BRD PWR', 'GPU 1 UTIL', 'CPU UTIL', 'GPU 2 PWR', etc.
    """
    if time_col not in df.columns:
        raise KeyError(f"time_col '{time_col}' not found; available: {df.columns.tolist()}")
    if y_col not in df.columns:
        raise KeyError(f"y_col '{y_col}' not found; available: {df.columns.tolist()}")

    plt.figure(figsize=(10, 4))
    plt.plot(df[time_col], df[y_col], lw=1.2)
    plt.grid(True, alpha=0.25)
    plt.xlabel("Time" if time_col == "timestamp" else time_col)
    plt.ylabel(ylabel or y_col)
    plt.title(title or y_col)
    plt.tight_layout()
    plt.show()




import numpy as np
import pandas as pd

def _to_seconds(col: pd.Series) -> np.ndarray:
    """Datetime → seconds since start; numeric stays numeric (shifted to start at 0)."""
    if np.issubdtype(col.dtype, np.datetime64):
        t0 = col.iloc[0]
        return (col - t0).dt.total_seconds().to_numpy(dtype=float)
    # numeric time (e.g., elapsed seconds)
    arr = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
    return arr - np.nanmin(arr)

def compute_idle_and_metrics(
    df: pd.DataFrame,
    y_col: str,
    *,
    time_col: str = "timestamp",
    idle_seconds: float = 1.0,          # mean of first N seconds
    idle_percentile: float | None = None,  # OR: use mean of bottom P% (e.g., 10.0)
    exclude_top_pct: float = 0.0,       # trim top P% outliers for stats (not for idle)
) -> dict:
    """
    Returns a dict with:
      idle_W, mean_W, std_W, min_W, max_W, p50_W, p90_W, p99_W,
      duration_s, n, dyn_mean_W  (mean above idle, clamped at 0)
    """
    if time_col not in df.columns:
        raise KeyError(f"{time_col!r} not in DataFrame")
    if y_col not in df.columns:
        raise KeyError(f"{y_col!r} not in DataFrame")

    # Clean & align
    t = _to_seconds(df[time_col])
    p = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(t) & np.isfinite(p)
    t, p = t[m], p[m]
    if t.size == 0:
        return {}

    # ---- Idle baseline ----
    idle_W = np.nan
    used_pct = None

    if idle_percentile is not None:
        # Mean of the bottom P% samples
        q = float(idle_percentile)
        cutoff = np.percentile(p, q)
        idle_W = float(np.mean(p[p <= cutoff])) if np.any(p <= cutoff) else float(np.mean(p))
        used_pct = q
    else:
        # Mean of first idle_seconds
        mask_idle = t <= (t.min() + float(idle_seconds))
        if np.any(mask_idle):
            idle_W = float(np.mean(p[mask_idle]))
        else:
            # Fallback: bottom 10% if the first-seconds window is empty
            cutoff = np.percentile(p, 10.0)
            idle_W = float(np.mean(p[p <= cutoff])) if np.any(p <= cutoff) else float(np.mean(p))
            used_pct = 10.0

    # ---- Stats (optionally trim top outliers) ----
    p_stats = p.copy()
    if exclude_top_pct > 0.0:
        cut = 100.0 - float(exclude_top_pct)
        thr = np.percentile(p_stats, cut)
        p_stats = p_stats[p_stats <= thr]

    # Basic stats
    mean_W = float(np.mean(p_stats))
    std_W  = float(np.std(p_stats, ddof=0))
    min_W  = float(np.min(p_stats))
    max_W  = float(np.max(p_stats))
    p50_W, p90_W, p99_W = [float(x) for x in np.percentile(p_stats, [50, 90, 99])]
    duration_s = float(t.max() - t.min()) if t.size > 1 else 0.0
    n = int(p.size)

    # Dynamic mean above idle (no energy, just average offset)
    dyn_mean_W = float(np.mean(np.maximum(0.0, p - idle_W))) if np.isfinite(idle_W) else float("nan")

    return {
        "column": y_col,
        "idle_W": idle_W,
        "idle_from": (f"first {idle_seconds}s" if used_pct is None else f"bottom {used_pct:.0f}%"),
        "mean_W": mean_W,
        "std_W": std_W,
        "min_W": min_W,
        "max_W": max_W,
        "p50_W": p50_W,
        "p90_W": p90_W,
        "p99_W": p99_W,
        "duration_s": duration_s,
        "n": n,
        "dyn_mean_W": dyn_mean_W,
        "exclude_top_pct": float(exclude_top_pct),
    }

def print_power_summary(stats: dict, label: str | None = None) -> None:
    if not stats:
        print("No data.")
        return
    tag = f"[{label}] " if label else ""
    print(f"{tag}{stats['column']}  n={stats['n']}  duration={stats['duration_s']:.3f}s")
    print(f"  idle ≈ {stats['idle_W']:.2f} W  ({stats['idle_from']})")
    print(f"  mean  {stats['mean_W']:.2f} W  ± {stats['std_W']:.2f}")
    print(f"  min/max {stats['min_W']:.2f}/{stats['max_W']:.2f} W")
    print(f"  P50/P90/P99  {stats['p50_W']:.2f}/{stats['p90_W']:.2f}/{stats['p99_W']:.2f} W")
    print(f"  dyn mean above idle ≈ {stats['dyn_mean_W']:.2f} W")

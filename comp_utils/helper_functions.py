import time
from typing import Dict, Any, Optional, Iterable

import numpy as np
import torch


def _try_import_dml():
    try:
        import torch_directml as dml  # type: ignore
        return dml
    except Exception:
        return None


def _resolve_device(device: str | torch.device) -> tuple[str, object]:
    """
    Returns (backend_tag, device_obj).
    backend_tag in {"cuda","mps","dml","cpu"}.
    """
    if isinstance(device, torch.device):
        tag = device.type
        return tag, device
    if isinstance(device, str):
        tag = device.lower()
        if tag.startswith("cuda"):
            return "cuda", torch.device(device)
        if tag.startswith("mps"):
            return "mps", torch.device("mps")
        if tag == "dml":
            dml = _try_import_dml()
            if dml is None:
                raise RuntimeError("torch-directml not installed; pip install torch-directml")
            return "dml", dml.device()
        return "cpu", torch.device("cpu")
    # fallback
    return "cpu", torch.device("cpu")


def _synchronize(tag: str, dev_obj: object) -> None:
    """Backend-specific synchronization to make timing accurate."""
    if tag == "cuda":
        torch.cuda.synchronize()
    elif tag == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif tag == "dml":
        dml = _try_import_dml()
        if dml is not None and hasattr(dml, "synchronize"):
            try:
                dml.synchronize()  # available on recent torch-directml
                return
            except Exception:
                pass
        # generic, always-works barrier: force tiny readback
        torch.tensor(0, device="cpu")  # ensure CPU context exists
        _ = torch.tensor(0, device=dev_obj).item()  # readback -> sync
    else:
        # CPU has no async queue
        pass


def _maybe_tqdm(it: Iterable, *, enabled: bool, desc: str):
    if not enabled:
        return it
    try:
        from tqdm.auto import tqdm
        return tqdm(it, desc=desc)
    except Exception:
        return it


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    mnist_loader: torch.utils.data.DataLoader,
    *,
    device: str | torch.device = "cpu",   # "cpu" | "cuda" | "mps" | "dml"
    amp: bool = False,                    # AMP only meaningful on CUDA here
    warmup_batches: int = 5,
    exclude_pct: float = 0.0,            # trim slowest P% batches
    progress: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate classification accuracy + latency/throughput with backend-aware sync.
    Works on CPU, CUDA, MPS, and DirectML (torch-directml).
    """
    tag, dev = _resolve_device(device)

    model.eval().to(dev)

    # Warm-up (don’t time)
    if warmup_batches > 0:
        it = iter(mnist_loader)
        for _ in range(warmup_batches):
            try:
                images, _ = next(it)
            except StopIteration:
                break
            images = images.to(dev, non_blocking=True)
            _ = model(images)
        _synchronize(tag, dev)

    # Prepare timing/acc buffers
    batch_ms: list[float] = []
    batch_sizes: list[int] = []
    total_correct = 0
    total_samples = 0

    # AMP only for CUDA in this simple helper
    if amp and tag == "cuda":
        amp_ctx = torch.autocast(device_type="cuda")
    else:
        from contextlib import nullcontext
        amp_ctx = nullcontext()

    for images, labels in _maybe_tqdm(mnist_loader, enabled=progress, desc="Eval"):
        images = images.to(dev, non_blocking=True)
        labels = labels.to(dev, non_blocking=True)

        _synchronize(tag, dev)            # make sure prior work finished
        t0 = time.perf_counter()

        with amp_ctx:
            logits = model(images)

        _synchronize(tag, dev)            # measure *only* this batch
        t1 = time.perf_counter()

        wall_ms = (t1 - t0) * 1e3
        batch_ms.append(wall_ms)
        bs = images.size(0)
        batch_sizes.append(bs)

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += bs

    # Per-sample latency and batch throughput arrays
    per_sample_ms = np.array([ms / bs for ms, bs in zip(batch_ms, batch_sizes)], dtype=np.float64)
    thr_ips_batch = np.array([bs / (ms / 1e3) for ms, bs in zip(batch_ms, batch_sizes)], dtype=np.float64)

    # Outlier trimming (optional)
    def _trim(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0 or exclude_pct <= 0.0:
            return arr
        cut = 100.0 - float(exclude_pct)
        thr = np.percentile(arr, cut)
        return arr[arr <= thr]

    per_sample_ms = _trim(per_sample_ms)
    thr_ips_batch = _trim(thr_ips_batch)

    # Stats helpers
    def stats(arr: Optional[np.ndarray]) -> dict:
        if arr is None or arr.size == 0:
            return {}
        p50, p90, p99 = np.percentile(arr, [50, 90, 99])
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p50": float(p50),
            "p90": float(p90),
            "p99": float(p99),
            "n": int(arr.size),
        }

    lat = stats(per_sample_ms)
    thr = stats(thr_ips_batch)

    total_wall_s = float(np.sum(np.array(batch_ms)) / 1e3)
    global_thr_ips = (total_samples / total_wall_s) if total_wall_s > 0 else float("inf")
    acc = total_correct / max(1, total_samples)

    # Print summary
    print(f"Backend            : {tag}")
    print(f"Accuracy           : {acc*100:.4f}%  ({total_correct}/{total_samples})")
    if lat:
        print(f"Latency / sample   : {lat['mean']:.4f} ms ± {lat['std']:.4f} "
              f"(min={lat['min']:.4f}, max={lat['max']:.4f}, "
              f"P50={lat['p50']:.4f}, P90={lat['p90']:.4f}, P99={lat['p99']:.4f}) "
              f"[batches={lat['n']}]")
    if thr:
        print(f"Throughput (batch) : {thr['mean']:.2f} ips ± {thr['std']:.2f} "
              f"(P50={thr['p50']:.2f}, P90={thr['p90']:.2f}, P99={thr['p99']:.2f})")
    print(f"Throughput (global): {global_thr_ips:.2f} ips "
          f"(total_time={total_wall_s:.3f}s, samples={total_samples})")

    return {
        "backend": tag,
        "accuracy": acc,
        "latency_ms_per_sample": lat,
        "throughput_ips_batchwise": thr,
        "throughput_ips_global": global_thr_ips,
        "n_samples": total_samples,
        "n_batches": len(batch_sizes),
    }




"""CPU inference benchmark: serialized size, single-window and batch latency."""

import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

from edgefit.config import BENCH_BATCH, BENCH_REPEATS, BENCH_WARMUP


def environment(device: str = "cpu") -> dict:
    """Timings are hardware-dependent, so always report them with the machine they came from."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or "n/a",
        "python": platform.python_version(),
        "logical_cpu_cores": os.cpu_count(),
        "torch_version": torch.__version__,
        "pytorch_training_device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }


def _single_thread(model):
    """Force single-threaded inference so all models are compared on one core."""
    params = model.get_params()
    model.set_params(**{k: 1 for k in params if k.endswith("n_jobs")})
    return model


def benchmark_model(
    name: str,
    model,
    X,
    out_dir: Path,
    warmup: int = BENCH_WARMUP,
    repeats: int = BENCH_REPEATS,
    batch: int = BENCH_BATCH,
) -> dict:
    """Serialize the model, then time warm single-window predictions (median) and one batch."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X = np.asarray(X, dtype=np.float32 if isinstance(model, nn.Module) else np.float64)[:batch]
    stem = name.replace(" ", "_").replace("(", "").replace(")", "")

    if isinstance(model, nn.Module):
        torch.set_num_threads(1)
        model = model.cpu().eval()
        path = out_dir / f"{stem}.pt"
        torch.save(model.state_dict(), path)
        X = torch.from_numpy(X)

        def run(x):
            with torch.inference_mode():
                return model(x).argmax(dim=1)
    else:
        model = _single_thread(model)
        path = out_dir / f"{stem}.joblib"
        joblib.dump(model, path)
        run = model.predict

    for i in range(warmup):
        run(X[i : i + 1])
    latencies = []
    for i in range(repeats):
        row = X[i % len(X) : i % len(X) + 1]
        t0 = time.perf_counter_ns()
        run(row)
        latencies.append(time.perf_counter_ns() - t0)
    t0 = time.perf_counter_ns()
    run(X)
    batch_ns = time.perf_counter_ns() - t0

    return {
        "model": name,
        "size_mb": path.stat().st_size / 1024**2,
        "single_window_latency_ms": float(np.median(latencies)) / 1e6,
        "batch_latency_ms": batch_ns / 1e6,
        "batch_size": len(X),
    }

"""A fixed test set saved to disk, so every method (matched filter, CNN) sees identical examples.

Layout: noise-only examples first, then n_per_snr signal examples at each integer SNR.
X is written with a memory map so the big array is never held twice in RAM.
"""
import json
from pathlib import Path

import numpy as np

from gwdetector.simulate import SimConfig, TEST_SEED, make_dataset

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "test_set"


def build_test_set(cfg, out_dir=DEFAULT_DIR, n_noise=20000, snrs=range(1, 11), n_per_snr=500,
                   seed=TEST_SEED):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = [(0, 0.0)] + [(k + 1, float(s)) for k, s in enumerate(snrs)]
    sizes = [n_noise] + [n_per_snr] * len(snrs)
    total = sum(sizes)
    X = np.lib.format.open_memmap(out_dir / "X.npy", mode="w+", dtype=np.float32,
                                  shape=(total, cfg.n))
    y, snr = np.zeros(total, dtype=int), np.zeros(total)
    meta = {k: np.zeros(total) for k in ("m1", "m2", "t_merger")}
    start = 0
    for (k, s), size in zip(groups, sizes):
        # every group gets its own seed derived from TEST_SEED -> independent noise
        Xg, yg, mg = make_dataset(cfg, size, seed + k, snr_range=(s, s),
                                  frac_signal=0.0 if k == 0 else 1.0)
        X[start:start + size], y[start:start + size], snr[start:start + size] = Xg, yg, mg["snr"]
        for name in meta:
            meta[name][start:start + size] = mg[name]
        start += size
    X.flush()
    np.savez(out_dir / "labels.npz", y=y, snr=snr, **meta)
    (out_dir / "config.json").write_text(json.dumps(
        dict(seed=seed, n_noise=n_noise, snrs=list(snrs), n_per_snr=n_per_snr,
             fs=cfg.fs, duration=cfg.duration, mass_range=cfg.mass_range,
             merger_range=cfg.merger_range)))
    return out_dir


def load_test_set(out_dir=DEFAULT_DIR):
    """X (memory-mapped, read-only), y, snr, meta dict."""
    out_dir = Path(out_dir)
    labels = np.load(out_dir / "labels.npz")
    X = np.load(out_dir / "X.npy", mmap_mode="r")
    meta = {k: labels[k] for k in ("m1", "m2", "t_merger")}
    return X, labels["y"], labels["snr"], meta

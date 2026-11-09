"""Tier 3, step 1: look at the simulated data (shapes, SNR check, example plot)."""
import matplotlib.pyplot as plt
import numpy as np

from gwdetector.data import FIG_DIR
from gwdetector.matched_filter import find_peak, snr_series
from gwdetector.simulate import (SimConfig, TEST_SEED, TRAIN_SEED, chirp_spectrum, make_dataset,
                                 make_fixed_snr_set, unit_chirp)

CFG = SimConfig()


def measured_peak_snr(X, y, meta, k=100):
    """Matched filter with the TRUE template: how large a peak do we actually measure?"""
    peaks = []
    for i in np.flatnonzero(y)[:k]:
        H = chirp_spectrum(CFG, meta["m1"][i], meta["m2"][i])
        snr = snr_series(np.fft.rfft(X[i].astype(float)), H, CFG.n)
        peaks.append(find_peak(snr, CFG.fs, edge=0.5)[0])
    return np.mean(peaks)


def main():
    X, y, meta = make_dataset(CFG, 2000, TRAIN_SEED)
    print(f"train: X{X.shape} {X.dtype}, signal fraction {y.mean():.3f}, "
          f"noise-only std {X[y == 0].std():.3f}")
    print(f"masses {meta['m1'].min():.0f}-{meta['m1'].max():.0f} Msun, "
          f"signal SNR range {meta['snr'][y == 1].min():.1f}-{meta['snr'][y == 1].max():.1f}")
    for snr in (6, 10, 15):
        Xs, ys, ms = make_fixed_snr_set(CFG, 200, snr, TEST_SEED)
        print(f"injected SNR {snr:2d} -> true-template matched filter peak "
              f"{measured_peak_snr(Xs, ys, ms):.1f} (noise-only peaks "
              f"{np.mean([find_peak(snr_series(np.fft.rfft(Xs[i].astype(float)), chirp_spectrum(CFG, ms['m1'][i], ms['m2'][i]), CFG.n), CFG.fs, 0.5)[0] for i in np.flatnonzero(ys == 0)[:100]]):.1f})")
    plot_examples()


def plot_examples():
    t = np.arange(CFG.n) / CFG.fs
    rng = np.random.default_rng(0)
    h = unit_chirp(CFG, 36.0, 29.0, 3.0)
    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    ax[0].plot(t, h * 20, lw=0.7, color="C3")
    ax[0].set_title("Noiseless chirp (m1=36, m2=29, scaled to SNR 20)")
    for a, snr in zip(ax[1:], (0, 8, 20)):
        a.plot(t, rng.normal(size=CFG.n) + snr * h, lw=0.5)
        a.set_title("Noise only" if snr == 0 else f"Chirp + white noise, SNR {snr}")
    ax[-1].set_xlabel("time [s]")
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "05_simulated_examples.png", dpi=120)


if __name__ == "__main__":
    main()

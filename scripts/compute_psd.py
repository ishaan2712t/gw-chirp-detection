"""Compute the noise PSD of H1 and L1 around GW150914, showing every step of Welch's method."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from gwdetector.data import DETECTORS, FIG_DIR, fetch_strain
from gwdetector.psd import welch_psd

EVENT = "GW150914"
SEGLEN = 4.0       # seconds per segment -> frequency resolution 1/4 = 0.25 Hz
OVERLAP = 0.5      # 50% overlap (standard for a Hann window)
COLORS = {"H1": "#d62728", "L1": "#1f77b4"}


def main():
    FIG_DIR.mkdir(exist_ok=True)
    strain = {ifo: fetch_strain(EVENT, ifo, duration=32) for ifo in DETECTORS}
    fs = strain["H1"].sample_rate.value

    results = {}
    for ifo, ts in strain.items():
        x = ts.value
        f, psd, steps = welch_psd(x, fs, SEGLEN, OVERLAP, average="median")
        results[ifo] = (f, psd, steps)

        # --- Cross-checks against library implementations -------------------
        nper = int(SEGLEN * fs)
        _, psd_scipy = welch(x, fs, window="hann", nperseg=nper, noverlap=nper // 2,
                             average="median")
        psd_gwpy = ts.psd(fftlength=SEGLEN, overlap=SEGLEN * OVERLAP,
                          window="hann", method="median").value
        band = (f > 20) & (f < 1500)
        print(f"\n{ifo}: {steps['segments'].shape[0]} segments x {nper} samples, "
              f"df = {f[1]:.3f} Hz, {len(f)} frequency bins")
        print(f"  max |ours/scipy - 1| (20-1500 Hz) = "
              f"{np.max(np.abs(psd[band] / psd_scipy[band] - 1)):.2e}")
        print(f"  max |ours/gwpy  - 1| (20-1500 Hz) = "
              f"{np.max(np.abs(psd[band] / psd_gwpy[band] - 1)):.2e}")
        # Parseval sanity check: integral of PSD ~ variance of the data
        var_psd = np.trapezoid(steps['periodograms'].mean(axis=0), f)
        print(f"  Parseval: integral PSD df = {var_psd:.3e}, var(x) = {x.var():.3e}")
        for f0 in (20, 50, 100, 200, 500, 1000):
            i = np.argmin(np.abs(f - f0))
            print(f"  ASD @ {f0:4d} Hz = {np.sqrt(psd[i]):.2e} /rtHz")

    plot_steps(results["H1"], fs, "H1")
    plot_final(results)


def plot_steps(result, fs, ifo):
    f, psd, steps = result
    segs, P, w = steps["segments"], steps["periodograms"], steps["window"]
    t = np.arange(segs.shape[1]) / fs
    step = int(SEGLEN * fs * (1 - OVERLAP)) / fs

    fig, ax = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    # Step 1: overlapping segments
    a = ax[0, 0]
    for k in range(segs.shape[0]):
        a.plot(t + k * step, segs[k] + 0 * k, lw=0.3, alpha=0.6)
        a.hlines(segs.min() * 1.1 - k * 0.12 * np.ptp(segs), k * step, k * step + SEGLEN,
                 color=f"C{k % 10}", lw=3)
    a.set_title(f"Step 1 — cut 32 s into {segs.shape[0]} × {SEGLEN:.0f} s segments, "
                f"{OVERLAP:.0%} overlap (bars)")
    a.set_xlabel("time [s]"); a.set_ylabel("strain"); a.set_yticks([])

    # Step 2: window one segment
    a = ax[0, 1]
    s0 = segs[0] - segs[0].mean()
    a.plot(t, s0, color="0.6", lw=0.5, label="segment 0 (mean removed)")
    a.plot(t, s0 * w, color=COLORS[ifo], lw=0.5, label="× Hann window")
    a.plot(t, w * np.abs(s0).max(), "k--", lw=1, label="Hann window (scaled)")
    a.set_title("Step 2 — taper each segment to zero at its edges (Hann window)")
    a.set_xlabel("time in segment [s]"); a.legend(loc="lower right", fontsize=8)

    # Step 3: FFT -> one periodogram; compare with no window
    a = ax[1, 0]
    from numpy.fft import rfft
    rect = 2 * np.abs(rfft(s0)) ** 2 / (fs * len(s0))
    a.loglog(f, np.sqrt(rect), color="0.6", lw=0.5, label="no window (spectral leakage)")
    a.loglog(f, np.sqrt(P[0]), color=COLORS[ifo], lw=0.5, label="Hann-windowed")
    a.set_xlim(10, 2048); a.set_ylim(1e-24, 1e-18)
    a.set_title("Step 3 — FFT one segment → periodogram |X(f)|² (noisy!)")
    a.set_xlabel("frequency [Hz]"); a.set_ylabel(r"$\sqrt{P(f)}$  [1/$\sqrt{Hz}$]")
    a.legend(fontsize=8)

    # Step 4: average many periodograms
    a = ax[1, 1]
    a.loglog(f, np.sqrt(P[0]), color="0.75", lw=0.5, label="single periodogram")
    a.loglog(f, np.sqrt(P.mean(axis=0)), color="C2", lw=0.7, label=f"mean of {len(P)}")
    a.loglog(f, np.sqrt(psd), color=COLORS[ifo], lw=0.9, label=f"median of {len(P)} (bias-corrected)")
    a.set_xlim(10, 2048); a.set_ylim(1e-24, 1e-18)
    a.set_title("Step 4 — average the periodograms → Welch PSD (variance ↓)")
    a.set_xlabel("frequency [Hz]"); a.legend(fontsize=8)

    fig.suptitle(f"Welch's method, step by step — {DETECTORS[ifo]} ({ifo}), {EVENT}")
    out = FIG_DIR / "02_psd_steps.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


def plot_final(results):
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for ifo, (f, psd, _) in results.items():
        ax.loglog(f, np.sqrt(psd), color=COLORS[ifo], lw=0.7, label=f"{DETECTORS[ifo]} ({ifo})")
    notes = [(20, "seismic /\nsuspension wall"), (36, "calibration\nlines"),
             (60, "60 Hz mains"), (250, "quantum shot noise\n(flat → rising)"),
             (500, "violin modes\n(~500 Hz)"), (1800, "anti-alias\nfilter roll-off")]
    for x, txt in notes:
        ax.annotate(txt, xy=(x, 6e-20), ha="center", va="bottom", fontsize=8, color="0.3")
    ax.set_xlim(10, 2048); ax.set_ylim(5e-24, 1e-19)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"ASD = $\sqrt{\mathrm{PSD}}$  [strain/$\sqrt{\mathrm{Hz}}$]")
    ax.set_title(f"Noise amplitude spectral density around {EVENT} "
                 f"(32 s, {SEGLEN:.0f} s Hann segments, median Welch)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out = FIG_DIR / "03_asd.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

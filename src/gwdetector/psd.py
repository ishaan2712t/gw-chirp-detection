"""Welch's method written out step by step (no black boxes)."""
import numpy as np
from scipy.signal import get_window


def segment(x, nperseg, noverlap):
    """Step 1: slice x into overlapping segments, shape (n_segments, nperseg)."""
    step = nperseg - noverlap
    n_seg = 1 + (len(x) - nperseg) // step
    return np.stack([x[i * step : i * step + nperseg] for i in range(n_seg)])


def periodograms(segments, fs, window="hann"):
    """Steps 2-4: detrend, window, FFT, and normalise each segment to a one-sided PSD."""
    nperseg = segments.shape[1]
    w = get_window(window, nperseg)
    segs = segments - segments.mean(axis=1, keepdims=True)    # remove DC offset
    X = np.fft.rfft(segs * w, axis=1)                         # windowed FFT
    # |X|^2 / (fs * sum(w^2)) gives units of strain^2/Hz and makes the
    # window's power loss cancel out (so white noise of variance s^2 -> s^2/(fs/2)).
    P = np.abs(X) ** 2 / (fs * np.sum(w**2))
    P[:, 1:] *= 2                                             # fold negative freqs in
    if nperseg % 2 == 0:
        P[:, -1] /= 2                                         # Nyquist bin isn't doubled
    freqs = np.fft.rfftfreq(nperseg, d=1 / fs)
    return freqs, P, w


def welch_psd(x, fs, seglen=4.0, overlap=0.5, window="hann", average="median"):
    """Full Welch estimate. Returns freqs, psd, plus the intermediates for plotting."""
    nperseg = int(seglen * fs)
    noverlap = int(overlap * nperseg)
    segs = segment(np.asarray(x), nperseg, noverlap)
    freqs, P, w = periodograms(segs, fs, window)
    if average == "mean":
        psd = P.mean(axis=0)
    else:
        # Median is robust to glitches / signals in a segment. For exponentially
        # distributed periodogram values median = ln(2) * mean, so correct the bias.
        # Exact finite-n factor (same as LALSuite / scipy): 1 - 1/2 + 1/3 - ... -> ln 2.
        n = P.shape[0]
        bias = 1 + sum(1 / (2 * k + 1) - 1 / (2 * k) for k in range(1, (n - 1) // 2 + 1))
        psd = np.median(P, axis=0) / bias
    return freqs, psd, {"segments": segs, "periodograms": P, "window": w}

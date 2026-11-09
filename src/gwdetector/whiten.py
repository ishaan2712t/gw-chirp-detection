"""Whitening: make the noise flat (same power at every frequency) and band-limit it."""
import numpy as np
from scipy.signal.windows import tukey

from gwdetector.psd import welch_psd


def band_weights(freqs, low, high, taper=2.0):
    """1 inside [low, high], 0 outside, with smooth cosine edges (sharp edges ring)."""
    rise = np.clip((freqs - low) / taper, 0, 1)
    fall = np.clip((high - freqs) / taper, 0, 1)
    w = np.minimum(rise, fall)
    return 0.5 * (1 - np.cos(np.pi * w))


def whiten_fd(X, freqs, psd_freqs, psd, fs, band=(20.0, 500.0)):
    """Whiten an rfft spectrum X: divide by sqrt(S(f)*fs/2), then zero outside `band`.

    The fs/2 factor makes white noise come out with unit variance per sample.
    Works for data and for templates, so both get identical treatment.
    """
    weights = band_weights(freqs, *band)
    S = np.interp(freqs, psd_freqs, psd)          # PSD is on a coarser grid than X
    scale = np.zeros_like(freqs)
    ok = weights > 0
    scale[ok] = weights[ok] / np.sqrt(S[ok] * fs / 2)
    return X * scale


def whitened_spectrum(x, fs, psd_freqs=None, psd=None, band=(20.0, 500.0), alpha=0.1):
    """Taper x, FFT it, whiten. If no PSD is given, estimate one from x itself."""
    if psd is None:
        psd_freqs, psd, _ = welch_psd(x, fs, seglen=4.0, overlap=0.5, average="median")
    X = np.fft.rfft(x * tukey(len(x), alpha))      # taper: the FFT assumes x repeats
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)
    return whiten_fd(X, freqs, psd_freqs, psd, fs, band)


def whiten(x, fs, **kwargs):
    """Whitened, band-passed time series (same length as x)."""
    return np.fft.irfft(whitened_spectrum(x, fs, **kwargs), n=len(x))

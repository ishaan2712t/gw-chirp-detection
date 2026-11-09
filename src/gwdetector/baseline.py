"""Matched-filter baseline for Tier 3: one detection score per example, from a template bank.

Speed trick: every template is band-limited well below fs/4, so only the lower half of the
spectrum is non-zero. Inverting that half gives the SNR series at every 2nd sample, exactly
(no approximation, just half the samples), at half the cost.
"""
import numpy as np
from scipy.fft import ifft, rfft

from gwdetector.matched_filter import template_norm
from gwdetector.simulate import chirp_spectrum


def template_matrix(cfg, params):
    """Unit-norm, half-band spectra of the bank templates, shape (B, n//2), complex64."""
    out = np.empty((len(params), cfg.n // 2), dtype=np.complex64)
    for j, (m1, m2) in enumerate(params):
        H = chirp_spectrum(cfg, m1, m2)
        out[j] = (H / template_norm(H, cfg.n))[: cfg.n // 2]
    return out


def snr_half_rate(Xhalf, Hhalf):
    """|SNR(t)| at every 2nd sample for data spectra (b, n//2) and one unit-norm template."""
    return np.abs(ifft(Xhalf * np.conj(Hhalf), axis=-1))


def bank_statistics(X, bank, cfg, windows, chunk=250):
    """Largest SNR over all templates and over times inside each (t_start, t_end) window.

    Returns stats (n_examples, n_windows) and the index of the winning template per window.
    """
    n_ex, half = len(X), cfg.n // 2
    slices = [slice(int(a * cfg.fs / 2), int(b * cfg.fs / 2)) for a, b in windows]
    stats = np.zeros((n_ex, len(windows)))
    best = np.zeros((n_ex, len(windows)), dtype=int)
    for s in range(0, n_ex, chunk):
        Xh = rfft(np.asarray(X[s:s + chunk], dtype=np.float32), axis=-1)[:, :half]
        Xh[:, 0] = 0                                  # DC carries no signal
        for j, H in enumerate(bank):
            snr = snr_half_rate(Xh, H)
            for w, sl in enumerate(slices):
                peak = snr[:, sl].max(axis=1)
                better = peak > stats[s:s + chunk, w]
                stats[s:s + chunk, w][better] = peak[better]
                best[s:s + chunk, w][better] = j
    return stats, best

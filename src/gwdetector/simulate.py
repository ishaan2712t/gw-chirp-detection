"""Tier 3 simulator: chirps injected into white Gaussian noise, with a known SNR.

White unit-variance noise means "already whitened", so the optimal SNR of a signal h is
simply sqrt(sum(h^2)). That makes it easy to dial in an exact SNR for each example.
"""
from dataclasses import dataclass

import numpy as np

from gwdetector.matched_filter import template_norm
from gwdetector.template import DEFAULT_CUT_FACTOR, isco_frequency, taylorf2
from gwdetector.whiten import band_weights

# Separate seeds so train / validation / test never share noise or chirp parameters.
TRAIN_SEED, VAL_SEED, TEST_SEED = 1001, 2002, 3003


@dataclass(frozen=True)
class SimConfig:
    fs: float = 2048.0                    # sample rate [Hz]
    duration: float = 4.0                 # seconds per example
    f_low: float = 20.0                   # chirp starts here [Hz]
    mass_range: tuple = (20.0, 50.0)      # each component mass drawn uniformly [Msun]
    merger_range: tuple = (2.2, 3.5)      # merger time [s]; early enough for the chirp to fit
    pn_order: int = 2
    cut_factor: float = DEFAULT_CUT_FACTOR

    @property
    def n(self):
        return int(self.fs * self.duration)


def chirp_spectrum(cfg, m1, m2):
    """rfft spectrum of the chirp (merger at t=0, arbitrary scale), smooth band edges.

    The matched-filter baseline must use this same function, otherwise it is filtering
    with a slightly different shape than the injected signal and loses SNR.
    """
    freqs = np.fft.rfftfreq(cfg.n, 1 / cfg.fs)
    H = taylorf2(freqs, m1, m2, cfg.f_low, cfg.pn_order, cfg.cut_factor)
    f_high = cfg.cut_factor * isco_frequency(m1, m2)
    return H * band_weights(freqs, cfg.f_low, f_high, taper=5.0)   # smooth edges: no ringing


def unit_chirp(cfg, m1, m2, t_merger):
    """Real chirp of length cfg.n with merger at t_merger, scaled to optimal SNR = 1."""
    H = chirp_spectrum(cfg, m1, m2)
    freqs = np.fft.rfftfreq(cfg.n, 1 / cfg.fs)
    H = H * np.exp(-2j * np.pi * freqs * t_merger)              # shift merger to t_merger
    return np.fft.irfft(H / template_norm(H, cfg.n), n=cfg.n)


def draw_parameters(cfg, rng, n):
    """Random (m1, m2, t_merger, phase) for n examples; m1 >= m2."""
    m = rng.uniform(*cfg.mass_range, size=(n, 2))
    return np.max(m, axis=1), np.min(m, axis=1), rng.uniform(*cfg.merger_range, size=n)


def make_dataset(cfg, n, seed, snr_range=(4.0, 20.0), frac_signal=0.5):
    """n examples: label 1 = chirp + noise (SNR ~ U(snr_range)), label 0 = noise only.

    Returns X (n, cfg.n) float32, y (n,) int, and a dict of the true parameters.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, cfg.n))
    y = (rng.random(n) < frac_signal).astype(int)
    m1, m2, tm = draw_parameters(cfg, rng, n)
    snr = np.where(y == 1, rng.uniform(*snr_range, size=n), 0.0)
    for i in np.flatnonzero(y):
        X[i] += snr[i] * unit_chirp(cfg, m1[i], m2[i], tm[i])
    meta = dict(m1=m1, m2=m2, t_merger=tm, snr=snr)
    return X.astype(np.float32), y, meta


def make_fixed_snr_set(cfg, n, snr, seed):
    """Test set where every signal has exactly `snr` (for efficiency-vs-SNR curves)."""
    return make_dataset(cfg, n, seed, snr_range=(snr, snr))

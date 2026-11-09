"""Matched filter: right normalisation on noise, right SNR and time for an injection."""
import numpy as np

from gwdetector.matched_filter import find_peak, snr_series, template_norm
from gwdetector.template import taylorf2
from gwdetector.whiten import whiten_fd

FS, N = 2048.0, 2048 * 64
FREQS = np.fft.rfftfreq(N, 1 / FS)
SIGMA = 1e-3
PSD_LEVEL = 2 * SIGMA**2 / FS                       # one-sided PSD of white noise
PSD = (FREQS, np.full_like(FREQS, PSD_LEVEL))


def _noise(seed):
    return np.random.default_rng(seed).normal(0, SIGMA, N)


def _whiten(X):
    return whiten_fd(X, FREQS, PSD[0], PSD[1], FS)


def _template_spectrum():
    return _whiten(taylorf2(FREQS, 30, 30, f_low=20))


def test_template_norm_equals_time_domain_norm():
    Hw = _template_spectrum()
    h = np.fft.irfft(Hw, n=N)
    assert abs(template_norm(Hw, N) / np.sqrt(np.sum(h**2)) - 1) < 1e-9


def test_pure_noise_has_unit_variance_real_part_and_rayleigh_mean():
    Hw = _template_spectrum()
    Xw = _whiten(np.fft.rfft(_noise(0)))
    snr = snr_series(Xw, Hw, N)
    # |z| of a unit complex Gaussian: mean sqrt(pi/2)=1.2533, rms sqrt(2)=1.4142
    assert abs(snr.mean() / 1.2533 - 1) < 0.03
    assert abs(np.sqrt(np.mean(snr**2)) / 1.4142 - 1) < 0.03


def test_injection_recovered_at_right_time_and_snr():
    Hw = _template_spectrum()
    h_w = np.fft.irfft(Hw, n=N)                     # whitened template, merger at t=0
    target = 20.0
    shift = 40000                                    # inject with merger at sample 40000
    amp = target / template_norm(Hw, N)
    signal_w = amp * np.roll(h_w, shift)
    Xw = _whiten(np.fft.rfft(_noise(1))) + np.fft.rfft(signal_w)
    value, idx = find_peak(snr_series(Xw, Hw, N), FS)
    assert idx == shift
    assert abs(value - target) < 3.0                 # noise adds ~ +-1 (Rician), allow 3


def test_snr_is_independent_of_signal_phase():
    Hw = _template_spectrum()
    amp = 15.0 / template_norm(Hw, N)
    peaks = []
    for phase in (0.0, 1.0, np.pi / 2):             # constant phase shift of the signal
        shifted = Hw * np.exp(1j * phase)
        Xw = amp * np.fft.rfft(np.fft.irfft(shifted, n=N))
        peaks.append(snr_series(Xw, Hw, N).max())
    assert np.ptp(peaks) < 0.2


def test_find_peak_ignores_edges():
    snr = np.zeros(10000)
    snr[10] = 100
    snr[5000] = 5
    assert find_peak(snr, fs=1000.0, edge=2.0) == (5.0, 5000)

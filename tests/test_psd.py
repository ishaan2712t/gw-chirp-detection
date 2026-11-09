"""Hand-written Welch PSD vs known answers and vs scipy."""
import numpy as np
from scipy.signal import welch

from gwdetector.psd import segment, welch_psd

FS = 1024.0
N64, N32 = 1024 * 64, 1024 * 32


def _white(n, sigma=2.0, seed=0):
    return np.random.default_rng(seed).normal(0, sigma, n)


def test_segment_shape_and_overlap():
    segs = segment(np.arange(100.0), nperseg=20, noverlap=10)
    assert segs.shape == (9, 20)
    assert segs[1, 0] == 10          # second segment starts half-way through the first


def test_white_noise_level_is_two_sigma2_over_fs():
    # one-sided PSD of white noise is 2*sigma^2/fs, flat in frequency
    f, psd, _ = welch_psd(_white(N64), FS, seglen=2.0, average="mean")
    expected = 2 * 2.0**2 / FS
    assert abs(psd[10:-10].mean() / expected - 1) < 0.02


def test_median_and_mean_agree_for_gaussian_noise():
    x = _white(N64)
    _, mean_psd, _ = welch_psd(x, FS, 2.0, average="mean")
    _, med_psd, _ = welch_psd(x, FS, 2.0, average="median")
    assert abs(med_psd[10:-10].mean() / mean_psd[10:-10].mean() - 1) < 0.03


def test_matches_scipy_median_welch():
    x = _white(N32, seed=3)
    n = int(2 * FS)
    f, ours, _ = welch_psd(x, FS, 2.0, 0.5, average="median")
    _, ref = welch(x, FS, window="hann", nperseg=n, noverlap=n // 2, average="median")
    np.testing.assert_allclose(ours, ref, rtol=1e-8)


def test_parseval_integral_equals_variance():
    x = _white(N64)
    f, psd, _ = welch_psd(x, FS, 2.0, average="mean")
    assert abs(np.trapezoid(psd, f) / x.var() - 1) < 0.03


def test_sine_peak_lands_in_right_bin():
    t = np.arange(int(FS * 16)) / FS
    f, psd, _ = welch_psd(np.sin(2 * np.pi * 100 * t), FS, 2.0, average="mean")
    assert abs(f[np.argmax(psd)] - 100) <= f[1]

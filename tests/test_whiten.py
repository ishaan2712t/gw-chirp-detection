"""Whitening should turn coloured noise into flat noise and respect the band."""
import numpy as np

from gwdetector.psd import welch_psd
from gwdetector.whiten import band_weights, whiten

FS = 2048.0
N = 2048 * 64


def _coloured_noise(seed=0):
    """Gaussian noise with a known, steeply sloped one-sided PSD S(f) = 1e-6*(1+(50/f)^4)."""
    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(N, 1 / FS)
    S = 1e-6 * (1 + (50 / np.maximum(f, 1.0)) ** 4)
    amp = np.sqrt(N * S * FS / 2)
    X = amp * (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)) / np.sqrt(2)
    return np.fft.irfft(X, n=N), f, S


def test_band_weights_shape():
    f = np.arange(0, 1000.0)
    w = band_weights(f, 20, 500)
    assert w[0] == 0 and w[10] == 0 and w[700] == 0   # outside
    assert w[100] == 1 and w[300] == 1                # inside
    assert 0 < w[21] < 1                              # taper


def test_whitened_noise_is_flat_at_expected_level():
    x, f, S = _coloured_noise()
    xw = whiten(x, FS, psd_freqs=f, psd=S)            # give the true PSD
    fw, psd_w, _ = welch_psd(xw[4096:-4096], FS, 4.0, average="mean")
    inband = (fw > 50) & (fw < 450)
    # unit-variance white noise has one-sided PSD 2/fs
    assert abs(psd_w[inband].mean() / (2 / FS) - 1) < 0.05
    assert psd_w[inband].std() / psd_w[inband].mean() < 0.3   # flat, not sloped


def test_out_of_band_power_is_removed():
    x, f, S = _coloured_noise(seed=1)
    xw = whiten(x, FS, psd_freqs=f, psd=S)
    fw, psd_w, _ = welch_psd(xw[4096:-4096], FS, 4.0, average="mean")
    assert psd_w[(fw > 700)].max() < 1e-3 * psd_w[(fw > 100) & (fw < 400)].mean()


def test_estimated_psd_also_whitens():
    x, _, _ = _coloured_noise(seed=2)
    xw = whiten(x, FS)                                # PSD estimated from the data
    fw, psd_w, _ = welch_psd(xw[4096:-4096], FS, 4.0, average="mean")
    inband = (fw > 50) & (fw < 450)
    assert abs(psd_w[inband].mean() / (2 / FS) - 1) < 0.10

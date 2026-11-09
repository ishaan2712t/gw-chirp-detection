"""Baseline matched filter: half-rate trick is exact, and the bank finds injected chirps."""
import numpy as np
from scipy.fft import rfft

from gwdetector.bank import build_bank, measure_mismatch
from gwdetector.baseline import bank_statistics, snr_half_rate, template_matrix
from gwdetector.matched_filter import snr_series, template_norm
from gwdetector.simulate import SimConfig, chirp_spectrum, make_fixed_snr_set

CFG = SimConfig()


def test_half_rate_equals_every_second_sample_of_full_filter():
    X, y, meta = make_fixed_snr_set(CFG, 10, snr=12.0, seed=4)
    i = int(np.flatnonzero(y)[0])
    H = chirp_spectrum(CFG, meta["m1"][i], meta["m2"][i])
    full = snr_series(np.fft.rfft(X[i].astype(float)), H, CFG.n)[::2]
    Hn = (H / template_norm(H, CFG.n))[: CFG.n // 2].astype(np.complex64)
    Xh = rfft(X[i].astype(np.float32))[: CFG.n // 2]
    Xh[0] = 0
    half = snr_half_rate(Xh[None], Hn)[0]
    np.testing.assert_allclose(half, full, rtol=1e-3, atol=2e-3)


def test_bank_has_expected_structure():
    bank = build_bank(CFG)
    assert 50 < len(bank) < 400
    assert np.all(bank[:, 0] >= bank[:, 1])                     # m1 >= m2
    assert bank.min() >= CFG.mass_range[0] - 2.0 - 1e-9 and bank.max() <= CFG.mass_range[1] + 2.0 + 1e-9


def test_finer_bank_has_larger_size_and_better_match():
    coarse, fine = build_bank(CFG, 0.04, 3), build_bank(CFG, 0.015, 5)
    assert len(fine) > len(coarse)
    assert measure_mismatch(CFG, fine, n=60).mean() > measure_mismatch(CFG, coarse, n=60).mean()


def test_match_is_at_most_one_and_high_for_default_bank():
    m = measure_mismatch(CFG, build_bank(CFG), n=100)
    assert m.max() < 1.01 and m.mean() > 0.99 and m.min() > 0.95


def test_noise_free_injection_scores_its_snr_and_exact_template_is_best():
    X, y, meta = make_fixed_snr_set(CFG, 8, snr=30.0, seed=6)
    i = int(np.flatnonzero(y)[0])
    params = np.array([[meta["m1"][i], meta["m2"][i]], [25.0, 25.0], [45.0, 30.0]])
    stats, best = bank_statistics(X[i:i + 1], template_matrix(CFG, params), CFG, [(0.5, 3.5)])
    assert best[0, 0] == 0                                      # the true template wins
    assert abs(stats[0, 0] - 30.0) < 3.0


def test_windows_restrict_the_peak_search():
    X, y, meta = make_fixed_snr_set(CFG, 6, snr=30.0, seed=7)
    i = int(np.flatnonzero(y)[0])
    params = np.array([[meta["m1"][i], meta["m2"][i]]])
    tm = meta["t_merger"][i]
    stats, _ = bank_statistics(X[i:i + 1], template_matrix(CFG, params), CFG,
                               [(tm - 0.2, tm + 0.2), (0.5, tm - 0.5)])
    assert stats[0, 0] > 25 and stats[0, 1] < 10                # window without the merger sees only noise

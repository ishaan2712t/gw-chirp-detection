"""The simulator must produce unit-variance white noise and signals with exactly the SNR we ask for."""
import numpy as np

from gwdetector.matched_filter import find_peak, snr_series
from gwdetector.simulate import (SimConfig, TEST_SEED, TRAIN_SEED, chirp_spectrum, make_dataset,
                                 make_fixed_snr_set, unit_chirp)

CFG = SimConfig()


def test_shapes_dtype_and_balance():
    X, y, meta = make_dataset(CFG, 400, seed=0)
    assert X.shape == (400, CFG.n) and X.dtype == np.float32 and y.shape == (400,)
    assert 0.4 < y.mean() < 0.6
    assert all(len(v) == 400 for v in meta.values())


def test_noise_only_examples_are_unit_white_gaussian():
    X, y, _ = make_dataset(CFG, 200, seed=1)
    noise = X[y == 0].astype(float)
    assert abs(noise.mean()) < 0.01 and abs(noise.std() - 1) < 0.01
    # white: neighbouring samples are uncorrelated
    assert abs(np.mean(noise[:, 1:] * noise[:, :-1])) < 0.01


def test_unit_chirp_has_unit_optimal_snr():
    h = unit_chirp(CFG, 36.0, 29.0, 3.0)
    assert abs(np.sqrt(np.sum(h**2)) - 1) < 1e-6


def test_chirp_fits_inside_the_window():
    # lowest-mass chirp is the longest; its early inspiral must not wrap round to the end
    h = unit_chirp(CFG, 20.0, 20.0, CFG.merger_range[0])
    edge = int(0.05 * CFG.fs)
    assert np.abs(h[:edge]).max() < 0.02 * np.abs(h).max()
    assert np.abs(h[-edge:]).max() < 0.02 * np.abs(h).max()


def test_merger_lands_at_requested_time():
    h = unit_chirp(CFG, 36.0, 29.0, 3.0)
    assert abs(np.argmax(np.abs(h)) / CFG.fs - 3.0) < 0.05


def test_matched_filter_measures_the_requested_snr():
    """End-to-end: matched filter with the true template should return ~ the injected SNR."""
    X, y, meta = make_fixed_snr_set(CFG, 60, snr=15.0, seed=5)
    peaks = []
    for i in np.flatnonzero(y)[:15]:
        H = chirp_spectrum(CFG, meta["m1"][i], meta["m2"][i])
        snr = snr_series(np.fft.rfft(X[i].astype(float)), H, CFG.n)
        v, idx = find_peak(snr, CFG.fs, edge=0.5)
        peaks.append(v)
        assert abs(idx / CFG.fs - meta["t_merger"][i]) < 0.02
    # E|z| for true SNR 15 is ~ sqrt(15^2 + 2) ~ 15.1; allow noise
    assert abs(np.mean(peaks) - 15.0) < 1.0


def test_same_seed_reproduces_and_train_test_seeds_differ():
    a, _, _ = make_dataset(CFG, 20, seed=7)
    b, _, _ = make_dataset(CFG, 20, seed=7)
    np.testing.assert_array_equal(a, b)
    tr, _, _ = make_dataset(CFG, 20, seed=TRAIN_SEED)
    te, _, _ = make_dataset(CFG, 20, seed=TEST_SEED)
    assert not np.allclose(tr, te)


def test_fixed_snr_set_has_exact_snr_labels():
    _, y, meta = make_fixed_snr_set(CFG, 50, snr=9.0, seed=2)
    assert np.all(meta["snr"][y == 1] == 9.0) and np.all(meta["snr"][y == 0] == 0.0)

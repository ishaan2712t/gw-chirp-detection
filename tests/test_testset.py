"""The saved test set must be reproducible and independent of the training seed."""
import numpy as np

from gwdetector.simulate import SimConfig, TRAIN_SEED, make_dataset
from gwdetector.testset import build_test_set, load_test_set

CFG = SimConfig(duration=2.0, merger_range=(1.2, 1.8))     # small for speed
KW = dict(n_noise=30, snrs=[2, 6], n_per_snr=10)


def test_layout_and_labels(tmp_path):
    d = build_test_set(CFG, tmp_path / "a", **KW)
    X, y, snr, meta = load_test_set(d)
    assert X.shape == (50, CFG.n) and X.dtype == np.float32
    assert y[:30].sum() == 0 and y[30:].all()
    assert set(snr[30:40]) == {2.0} and set(snr[40:]) == {6.0} and snr[:30].sum() == 0
    assert (d / "config.json").exists()


def test_rebuilding_gives_identical_data(tmp_path):
    a = load_test_set(build_test_set(CFG, tmp_path / "a", **KW))[0]
    b = load_test_set(build_test_set(CFG, tmp_path / "b", **KW))[0]
    np.testing.assert_array_equal(a, b)


def test_test_set_differs_from_training_data(tmp_path):
    X = load_test_set(build_test_set(CFG, tmp_path / "a", **KW))[0]
    train, _, _ = make_dataset(CFG, 50, TRAIN_SEED)
    assert not np.any(np.isclose(X[0], train[0]))
    assert abs(np.corrcoef(np.asarray(X[:30]).ravel(), train[:30].ravel())[0, 1]) < 0.02


def test_noise_groups_are_independent_across_snr_groups(tmp_path):
    X = np.asarray(load_test_set(build_test_set(CFG, tmp_path / "a", **KW))[0])
    assert abs(np.corrcoef(X[30:40].ravel(), X[40:50].ravel())[0, 1]) < 0.02

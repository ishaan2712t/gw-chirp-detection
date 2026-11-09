"""Metrics checked against Gaussian cases with known analytic answers."""
import numpy as np
from scipy.stats import norm

from gwdetector.metrics import auc, efficiency_at_fap, efficiency_error, roc_curve, threshold_at_fap

RNG = np.random.default_rng(0)
NEG = RNG.normal(0, 1, 200_000)
POS = RNG.normal(3, 1, 200_000)


def test_threshold_matches_gaussian_quantile():
    for fap in (1e-1, 1e-2):
        assert abs(threshold_at_fap(NEG, fap) - norm.isf(fap)) < 0.03


def test_efficiency_matches_analytic_value():
    # detect if score > isf(fap); signal ~ N(3,1)  ->  P = sf(isf(fap) - 3)
    for fap in (1e-1, 1e-2, 1e-3):
        expected = norm.sf(norm.isf(fap) - 3)
        assert abs(efficiency_at_fap(NEG, POS, fap) - expected) < 0.01


def test_roc_endpoints_and_monotone():
    fpr, tpr = roc_curve(NEG[:5000], POS[:5000])
    assert fpr[0] == 1 and tpr[0] == 1 and fpr[-1] == 0 and tpr[-1] == 0
    assert np.all(np.diff(fpr) <= 0) and np.all(np.diff(tpr) <= 0)


def test_auc_known_values():
    fpr, tpr = roc_curve(NEG, POS)
    assert abs(auc(fpr, tpr) - norm.cdf(3 / np.sqrt(2))) < 0.002    # equal-variance Gaussians
    fpr, tpr = roc_curve(NEG[:5000], NEG[5000:10000])
    assert abs(auc(fpr, tpr) - 0.5) < 0.03                          # no information


def test_tiny_example_by_hand():
    neg, pos = np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.5, 5.0])
    assert threshold_at_fap(neg, 0.25) == 3.0      # one of four noise values may exceed it
    assert efficiency_at_fap(neg, pos, 0.25) == 0.5


def test_efficiency_error():
    assert abs(efficiency_error(0.5, 100) - 0.05) < 1e-12

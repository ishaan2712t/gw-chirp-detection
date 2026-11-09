"""Detection metrics computed from scores: ROC curve and efficiency at a fixed false-alarm probability."""
import numpy as np


def roc_curve(neg_scores, pos_scores):
    """False-positive and true-positive rates for every threshold (score > threshold = detect).

    neg_scores: scores of noise-only examples. pos_scores: scores of signal examples.
    """
    neg, pos = np.sort(neg_scores), np.sort(pos_scores)
    thresholds = np.concatenate(([-np.inf], neg, [np.inf]))
    fpr = 1 - np.searchsorted(neg, thresholds, side="right") / len(neg)
    tpr = 1 - np.searchsorted(pos, thresholds, side="right") / len(pos)
    return fpr, tpr


def threshold_at_fap(neg_scores, fap):
    """Smallest threshold whose false-alarm probability on noise is <= fap."""
    neg = np.sort(neg_scores)
    k = int(np.floor(fap * len(neg)))            # how many noise examples may exceed it
    return neg[-k - 1] if k < len(neg) else -np.inf


def efficiency_at_fap(neg_scores, pos_scores, fap):
    """Fraction of signal examples above the threshold that noise exceeds with probability fap."""
    return float(np.mean(np.asarray(pos_scores) > threshold_at_fap(neg_scores, fap)))


def efficiency_error(eff, n):
    """Binomial standard error of an efficiency measured on n examples."""
    return float(np.sqrt(eff * (1 - eff) / n))


def auc(fpr, tpr):
    """Area under the ROC curve (fpr decreases as the threshold rises, so integrate backwards)."""
    return float(-np.trapezoid(tpr, fpr))

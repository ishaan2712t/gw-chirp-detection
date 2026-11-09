"""Tier 3: matched-filter baseline on the fixed test set -> ROC and efficiency curves."""
import json

import matplotlib.pyplot as plt
import numpy as np

from gwdetector.bank import build_bank, measure_mismatch
from gwdetector.baseline import bank_statistics, template_matrix
from gwdetector.data import FIG_DIR, PROJECT_ROOT
from gwdetector.metrics import auc, efficiency_at_fap, efficiency_error, roc_curve
from gwdetector.simulate import SimConfig
from gwdetector.testset import load_test_set

CFG = SimConfig()
ROC_SNRS = (2, 4, 6)
FAPS = (1e-1, 1e-2, 1e-3)
RESULTS = PROJECT_ROOT / "results"


def main():
    X, y, snr, _ = load_test_set()
    params = build_bank(CFG)
    mism = 1 - measure_mismatch(CFG, params, n=2000, seed=99)     # fresh random injections
    print(f"bank: {len(params)} templates; mismatch over 2000 random injections: "
          f"mean {mism.mean():.4f}, 95th pct {np.percentile(mism, 95):.4f}, max {mism.max():.4f}")
    # window A: whole segment minus 0.5 s edges. window B: only where mergers can occur.
    windows = [(0.5, CFG.duration - 0.5), (CFG.merger_range[0] - 0.1, CFG.merger_range[1] + 0.1)]
    stats, _ = bank_statistics(X, template_matrix(CFG, params), CFG, windows)
    RESULTS.mkdir(exist_ok=True)
    np.savez(RESULTS / "mf_scores.npz", full=stats[:, 0], prior=stats[:, 1], y=y, snr=snr)
    out = dict(bank_size=len(params), mismatch_mean=float(mism.mean()),
               mismatch_p95=float(np.percentile(mism, 95)), mismatch_max=float(mism.max()),
               n_noise=int((y == 0).sum()), n_per_snr=int((snr == 1).sum()))
    for name, col in (("full", 0), ("prior", 1)):
        out[name] = summarise(stats[:, col], y, snr)
    (RESULTS / "baseline_mf.json").write_text(json.dumps(out, indent=2))
    plot(stats[:, 0], y, snr)
    report(out)


def summarise(score, y, snr):
    neg = score[y == 0]
    res = {"auc": {}, "efficiency": {}}
    for s in ROC_SNRS:
        res["auc"][s] = auc(*roc_curve(neg, score[snr == s]))
    for fap in FAPS:
        res["efficiency"][str(fap)] = {
            str(s): efficiency_at_fap(neg, score[snr == s], fap) for s in range(1, 11)}
    return res


def plot(score, y, snr):
    neg = score[y == 0]
    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    for s in ROC_SNRS:
        fpr, tpr = roc_curve(neg, score[snr == s])
        ax.plot(fpr, tpr, label=f"SNR {s}")
    ax.plot([1e-4, 1], [1e-4, 1], "k:", lw=0.8, label="chance")
    ax.set(xscale="log", xlim=(5e-5, 1), ylim=(0, 1), xlabel="false-alarm probability",
           ylabel="true-positive rate", title="Matched filter ROC (template bank)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(FIG_DIR / "06_roc_matched_filter.png", dpi=130)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    snrs = np.arange(1, 11)
    n = int((snr == 1).sum())
    for fap in FAPS:
        eff = np.array([efficiency_at_fap(neg, score[snr == s], fap) for s in snrs])
        err = [efficiency_error(e, n) for e in eff]
        ax.errorbar(snrs, eff, yerr=err, marker="o", ms=4, capsize=2, label=f"FAP = {fap:g}")
    ax.set(xlabel="injected (optimal) SNR", ylabel="detection efficiency", ylim=(0, 1.02),
           title="Matched filter efficiency vs SNR")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(FIG_DIR / "07_efficiency_matched_filter.png", dpi=130)


def report(out):
    print("AUC at SNR 2/4/6 (full window): " + " / ".join(f"{out['full']['auc'][s]:.3f}" for s in ROC_SNRS))
    for fap in FAPS:
        e = out["full"]["efficiency"][str(fap)]
        print(f"efficiency @FAP {fap:g}, SNR 2/4/6/8: " + " / ".join(f"{e[str(s)]:.3f}" for s in (2, 4, 6, 8)))
    eprior = out["prior"]["efficiency"]["0.001"]
    print(f"(merger-window variant, FAP 1e-3, SNR 4/6: {eprior['4']:.3f} / {eprior['6']:.3f})")


if __name__ == "__main__":
    main()

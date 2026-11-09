"""Score the trained CNN on the fixed test set and compare it with the matched filter."""
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from gwdetector.cnn import ChirpCNN, predict_scores
from gwdetector.data import FIG_DIR, PROJECT_ROOT
from gwdetector.metrics import auc, efficiency_at_fap, efficiency_error, roc_curve
from gwdetector.testset import load_test_set

RESULTS = PROJECT_ROOT / "results"
ROC_SNRS, FAPS, SNRS = (2, 4, 6), (1e-1, 1e-2, 1e-3), np.arange(1, 11)
METHODS = {"matched filter (bank)": "#1f77b4", "CNN": "#d62728"}


def cnn_scores(X):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChirpCNN().to(device)
    model.load_state_dict(torch.load(RESULTS / "cnn_best.pt", map_location=device))
    return predict_scores(model, X, device)


def table(score, y, snr):
    neg = score[y == 0]
    return dict(
        auc={s: auc(*roc_curve(neg, score[snr == s])) for s in ROC_SNRS},
        efficiency={str(f): {int(s): efficiency_at_fap(neg, score[snr == s], f) for s in SNRS}
                    for f in FAPS})


def plot_roc(scores, y, snr):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True, constrained_layout=True)
    for ax, s in zip(axes, ROC_SNRS):
        for (name, col), sc in zip(METHODS.items(), scores):
            fpr, tpr = roc_curve(sc[y == 0], sc[snr == s])
            ax.plot(fpr, tpr, color=col, label=name)
        ax.plot([5e-5, 1], [5e-5, 1], "k:", lw=0.8)
        ax.set(xscale="log", xlim=(5e-5, 1), ylim=(0, 1), title=f"injected SNR {s}",
               xlabel="false-alarm probability")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("true-positive rate"); axes[0].legend(loc="lower right")
    fig.savefig(FIG_DIR / "08_roc_cnn_vs_mf.png", dpi=130)


def plot_eff(scores, y, snr):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True, constrained_layout=True)
    n = int((snr == 1).sum())
    for ax, fap in zip(axes, FAPS):
        for (name, col), sc in zip(METHODS.items(), scores):
            eff = np.array([efficiency_at_fap(sc[y == 0], sc[snr == s], fap) for s in SNRS])
            ax.errorbar(SNRS, eff, yerr=[efficiency_error(e, n) for e in eff], color=col,
                        marker="o", ms=4, capsize=2, label=name)
        ax.set(title=f"FAP = {fap:g}", xlabel="injected (optimal) SNR", ylim=(0, 1.02))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("detection efficiency"); axes[0].legend(loc="upper left")
    fig.savefig(FIG_DIR / "09_efficiency_cnn_vs_mf.png", dpi=130)


def main():
    X, y, snr, _ = load_test_set()
    mf = np.load(RESULTS / "mf_scores.npz")
    assert np.array_equal(mf["y"], y) and np.array_equal(mf["snr"], snr), "different test set!"
    cnn = cnn_scores(X)
    np.savez(RESULTS / "cnn_scores.npz", score=cnn, y=y, snr=snr)
    out = {"matched_filter": table(mf["full"], y, snr), "matched_filter_merger_window": table(mf["prior"], y, snr),
           "cnn": table(cnn, y, snr)}
    (RESULTS / "comparison_cnn_vs_mf.json").write_text(json.dumps(out, indent=2))
    plot_roc((mf["full"], cnn), y, snr)
    plot_eff((mf["full"], cnn), y, snr)
    for name, key in (("MF full-window", "matched_filter"), ("MF merger-window", "matched_filter_merger_window"), ("CNN", "cnn")):
        t = out[key]
        print(f"{name:17s} AUC 2/4/6: " + "/".join(f"{t['auc'][s]:.3f}" for s in ROC_SNRS))
        for f in FAPS:
            e = t["efficiency"][str(f)]
            print(f"{'':17s} eff@FAP {f:g}, SNR 2/4/6/8: " + "/".join(f"{e[s]:.3f}" for s in (2, 4, 6, 8)))


if __name__ == "__main__":
    main()

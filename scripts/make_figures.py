"""Final figures from saved results: efficiency vs SNR with bootstrap bands, and the ROC at SNR 6."""
import csv

import matplotlib.pyplot as plt
import numpy as np

from gwdetector.data import FIG_DIR, PROJECT_ROOT
from gwdetector.metrics import roc_curve

RESULTS = PROJECT_ROOT / "results"
STYLE = {"matched_filter": ("matched filter", "k"), "base": ("CNN base (40 epochs)", "#1f77b4"),
         "wide": ("CNN wide (4x params)", "#2ca02c"), "long": ("CNN long (120 epochs)", "#d62728")}


def read_rows(kind):
    with open(RESULTS / "bootstrap_ci.csv") as f:
        return [r for r in csv.DictReader(f) if r["kind"] == kind]


def efficiency_figure():
    mf = {int(r["snr"]): r for r in read_rows("efficiency") if r["model"] == "matched_filter"}
    single = read_rows("efficiency_single_run_mean")
    fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
    series = {"matched_filter": mf, **{v: {int(r["snr"]): r for r in single if r["model"] == v} for v in ("base", "wide", "long")}}
    for m, rows in series.items():
        label, color = STYLE[m]
        snr = [s for s in sorted(rows) if 2 <= s <= 8]
        ax.plot(snr, [float(rows[s]["value"]) for s in snr], "-o", ms=3, color=color, label=label)
        ax.fill_between(snr, [float(rows[s]["ci_lo"]) for s in snr], [float(rows[s]["ci_hi"]) for s in snr],
                        color=color, alpha=0.18, lw=0)
    ax.set(xlabel="injected optimal SNR (white noise, unit variance)", ylabel="detection efficiency at FAP = 0.01",
           title="Efficiency vs SNR (CNN lines: mean of 5 runs; bands: 95% bootstrap CI)")
    ax.title.set_fontsize(9)
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8)
    fig.savefig(FIG_DIR / "10_efficiency_vs_snr_ci.png", dpi=200)


def mean_roc(runs, y, snr, s, grid):
    tprs = []
    for sc in runs:
        fpr, tpr = roc_curve(sc[y == 0], sc[snr == s])
        tprs.append(np.interp(grid, fpr[::-1], tpr[::-1]))      # fpr decreases along the arrays
    return np.mean(tprs, axis=0)


def roc_figure(s=6):
    ref = np.load(RESULTS / "mf_scores.npz")
    y, snr = ref["y"], ref["snr"]
    grid = np.logspace(-4, 0, 400)
    fig, ax = plt.subplots(figsize=(6, 4.8), constrained_layout=True)
    ax.plot(grid, mean_roc([ref["full"]], y, snr, s, grid), color="k", label=STYLE["matched_filter"][0])
    for v in ("base", "wide", "long"):
        runs = [np.load(f) for f in sorted((RESULTS / "runs").glob(f"{v}_s*_scores.npy"))]
        ax.plot(grid, mean_roc(runs, y, snr, s, grid), color=STYLE[v][1], label=STYLE[v][0])
    ax.axvline(0.01, color="0.6", ls="--", lw=0.8)
    ax.set(xscale="log", xlabel="false-alarm probability (noise-only fraction above threshold)",
           ylabel="detection probability (true-positive rate)", title=f"ROC at injected SNR {s} (CNN: TPR averaged over 5 runs)")
    ax.title.set_fontsize(9)
    ax.grid(alpha=0.3, which="both"); ax.legend(loc="lower right", fontsize=8)
    fig.savefig(FIG_DIR / "11_roc_snr6.png", dpi=200)


if __name__ == "__main__":
    efficiency_figure(); roc_figure()
    print("saved figures 10, 11")

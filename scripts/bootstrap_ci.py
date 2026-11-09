"""Bootstrap confidence intervals for efficiency at FAP 0.01, from the saved scores only (no retraining).

Each replicate resamples the 20,000 noise examples and the 500 signals of every SNR with replacement,
re-derives the threshold from the resampled noise, and recomputes every model's efficiency on the
SAME resample, so differences between models are paired.
CNN variants are pooled by averaging the 5 seeds' scores per example (an ensemble); the seed
std of the single-run efficiencies is reported separately.
"""
import csv

import numpy as np

from gwdetector.data import PROJECT_ROOT
from gwdetector.metrics import efficiency_at_fap

RESULTS = PROJECT_ROOT / "results"
FAP, N_BOOT, SEED = 0.01, 2000, 12345
SNRS = list(range(1, 11))
VARIANTS = ("base", "wide", "long")


def load_models():
    ref = np.load(RESULTS / "mf_scores.npz")
    y, snr = ref["y"], ref["snr"]
    scores = {"matched_filter": ref["full"]}
    seeds = {}
    for v in VARIANTS:
        runs = [np.load(f) for f in sorted((RESULTS / "runs").glob(f"{v}_s*_scores.npy"))]
        scores[v] = np.mean(runs, axis=0)           # pooled over seeds, example by example
        seeds[v] = runs
        for i, r in enumerate(runs):
            scores[f"{v}_seed{i}"] = r              # also bootstrapped, for the single-run mean
    return scores, seeds, y, snr


def split(score, y, snr):
    return score[y == 0], {s: score[snr == s] for s in SNRS}


def bootstrap(scores, y, snr, rng):
    """Returns eff[model][snr] -> array (N_BOOT,), all models sharing each resample."""
    parts = {m: split(sc, y, snr) for m, sc in scores.items()}
    n_neg = len(parts["matched_filter"][0])
    n_pos = len(parts["matched_filter"][1][SNRS[0]])
    k = int(np.floor(FAP * n_neg))
    eff = {m: {s: np.empty(N_BOOT) for s in SNRS} for m in scores}
    for b in range(N_BOOT):
        ni = rng.integers(0, n_neg, n_neg)
        pi = {s: rng.integers(0, n_pos, n_pos) for s in SNRS}
        for m, (neg, pos) in parts.items():
            thr = np.sort(neg[ni])[-k - 1]
            for s in SNRS:
                eff[m][s][b] = np.mean(pos[s][pi[s]] > thr)
    return eff


def ci(a):
    return np.percentile(a, [2.5, 97.5])


def main():
    scores, seeds, y, snr = load_models()
    eff = bootstrap(scores, y, snr, np.random.default_rng(SEED))
    # "<v>_single" = mean efficiency of the individual runs (what one trained network achieves)
    for v in VARIANTS:
        eff[f"{v}_single"] = {s: np.mean([eff[f"{v}_seed{i}"][s] for i in range(len(seeds[v]))], axis=0)
                              for s in SNRS}
    rows = []
    for m, sc in scores.items():
        if "_seed" in m:
            continue
        neg, pos = split(sc, y, snr)
        for s in SNRS:
            lo, hi = ci(eff[m][s])
            std = ""
            if m in seeds:     # seed-to-seed std of the single-run efficiencies
                single = [efficiency_at_fap(*[split(r, y, snr)[0], split(r, y, snr)[1][s]], FAP) for r in seeds[m]]
                std = f"{np.std(single, ddof=1):.4f}"
            rows.append(dict(kind="efficiency", model=m, snr=s, value=f"{efficiency_at_fap(neg, pos[s], FAP):.4f}",
                             ci_lo=f"{lo:.4f}", ci_hi=f"{hi:.4f}", seed_std=std, distinguishable_from_0=""))
    for v in VARIANTS:
        lo, hi = None, None
        for s in SNRS:
            point = np.mean([efficiency_at_fap(*[split(r, y, snr)[0], split(r, y, snr)[1][s]], FAP) for r in seeds[v]])
            lo, hi = ci(eff[f"{v}_single"][s])
            rows.append(dict(kind="efficiency_single_run_mean", model=v, snr=s, value=f"{point:.4f}",
                             ci_lo=f"{lo:.4f}", ci_hi=f"{hi:.4f}", seed_std="", distinguishable_from_0=""))
    for m in [*VARIANTS, *(f"{v}_single" for v in VARIANTS)]:
        for s in SNRS:
            d = eff["matched_filter"][s] - eff[m][s]
            lo, hi = ci(d)
            kind = "efficiency_single_run_mean" if m.endswith("_single") else "efficiency"
            point = rows_value(rows, "efficiency", "matched_filter", s) - rows_value(rows, kind, m.replace("_single", ""), s)
            rows.append(dict(kind="paired_diff_MF_minus_" + ("single_run_mean" if m.endswith("_single") else "pooled"),
                             model=m.replace("_single", ""), snr=s, value=f"{point:.4f}",
                             ci_lo=f"{lo:.4f}", ci_hi=f"{hi:.4f}", seed_std="",
                             distinguishable_from_0="yes" if lo > 0 or hi < 0 else "no"))
    # Does longer training / a wider network change what a single run achieves? (same resamples)
    for a in ("long", "wide"):
        for s in SNRS:
            d = eff[f"{a}_single"][s] - eff["base_single"][s]
            lo, hi = ci(d)
            point = rows_value(rows, "efficiency_single_run_mean", a, s) - rows_value(rows, "efficiency_single_run_mean", "base", s)
            rows.append(dict(kind="paired_diff_single_run_mean", model=f"{a}_minus_base", snr=s, value=f"{point:.4f}",
                             ci_lo=f"{lo:.4f}", ci_hi=f"{hi:.4f}", seed_std="",
                             distinguishable_from_0="yes" if lo > 0 or hi < 0 else "no"))
    out = RESULTS / "bootstrap_ci.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{N_BOOT} resamples, FAP {FAP}, wrote {out.name}")
    for r in rows:
        if r["snr"] in (4, 6, 8) and r["kind"] != "efficiency":
            print(f"{r['kind'][-18:]:18s} {r['model']:15s} snr{r['snr']} {r['value']} [{r['ci_lo']},{r['ci_hi']}] "
                  f"{r['seed_std']} {r['distinguishable_from_0']}")


def rows_value(rows, kind, model, s):
    return next(float(r["value"]) for r in rows if r["kind"] == kind and r["model"] == model and r["snr"] == s)


if __name__ == "__main__":
    main()

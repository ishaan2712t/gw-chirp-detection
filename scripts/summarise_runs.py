"""Mean +- std over seeds of efficiency (and AUC) for each variant, plus the matched filter."""
import json

import numpy as np

from gwdetector.data import PROJECT_ROOT
from gwdetector.metrics import auc, efficiency_at_fap, roc_curve

RESULTS = PROJECT_ROOT / "results"
RUNS = RESULTS / "runs"
SNRS, FAPS = (4, 6, 8), (1e-1, 1e-2, 1e-3)
AUC_SNRS = (2, 4, 6, 8)


def metrics(score, y, snr):
    neg = score[y == 0]
    m = {f"eff_{f:g}_snr{s}": efficiency_at_fap(neg, score[snr == s], f) for f in FAPS for s in SNRS}
    m.update({f"auc_snr{s}": auc(*roc_curve(neg, score[snr == s])) for s in AUC_SNRS})
    return m


def main():
    ref = np.load(RESULTS / "mf_scores.npz")
    y, snr = ref["y"], ref["snr"]
    out = {"n_noise_only": int((y == 0).sum()), "n_per_snr": int((snr == 8).sum()),
           "matched_filter": metrics(ref["full"], y, snr), "variants": {}}
    for variant in ("base", "wide", "long"):
        files = sorted(RUNS.glob(f"{variant}_s*_scores.npy"))
        if not files:
            continue
        per_seed = [metrics(np.load(f), y, snr) for f in files]
        info = [json.loads(f.with_name(f.name.replace("_scores.npy", ".json")).read_text()) for f in files]
        agg = {k: (float(np.mean([m[k] for m in per_seed])), float(np.std([m[k] for m in per_seed], ddof=1))
                   if len(per_seed) > 1 else 0.0) for k in per_seed[0]}
        out["variants"][variant] = dict(
            n_seeds=len(files), params=info[0]["params"], epochs=info[0]["epochs"],
            mean_train_seconds=float(np.mean([i["seconds"] for i in info])),
            mean_best_epoch=float(np.mean([i["best_epoch"] for i in info])),
            per_seed=per_seed, mean_std=agg)
    (RESULTS / "seed_variation.json").write_text(json.dumps(out, indent=2))
    mf = out["matched_filter"]
    print(f"test set: {out['n_noise_only']} noise-only examples, {out['n_per_snr']} signals per SNR")
    print("efficiency at FAP 0.01 (mean +- std over seeds)   SNR 4 / 6 / 8")
    print(f"  {'matched filter':15s} " + " / ".join(f"{mf[f'eff_0.01_snr{s}']:.3f}" for s in SNRS) + "   (one deterministic value)")
    for v, d in out["variants"].items():
        a = d["mean_std"]
        print(f"  {v + f' (n={d['n_seeds']})':15s} " + " / ".join(
            f"{a[f'eff_0.01_snr{s}'][0]:.3f}+-{a[f'eff_0.01_snr{s}'][1]:.3f}" for s in SNRS)
            + f"   {d['params']:,} params, {d['mean_train_seconds']:.0f}s/run")
    print("AUC at SNR 2/4/6/8:  MF " + "/".join(f"{mf[f'auc_snr{s}']:.3f}" for s in AUC_SNRS))


if __name__ == "__main__":
    main()

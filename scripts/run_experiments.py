"""Repeat CNN training over several seeds for several variants; log to a file.

Each finished run saves its test-set scores (results/runs/<variant>_s<seed>_scores.npy), so the
script can be stopped and restarted: finished runs are skipped.
"""
import argparse
import json
import time

import numpy as np

from gwdetector.data import PROJECT_ROOT
from gwdetector.simulate import SimConfig
from gwdetector.testset import load_test_set
# torch and the CNN code are imported inside the functions below: on Windows every data-generation
# worker re-imports this file, and importing torch in each of them wastes a lot of memory.

RUNS = PROJECT_ROOT / "results" / "runs"
VARIANTS = {                       # name: (epochs, width)
    "base": (40, 1),               # the configuration used so far
    "wide": (40, 2),               # every channel count doubled
    "long": (120, 1),              # 3x the epochs (and 3x the fresh training examples)
}


def run_one(variant, seed, X, log):
    import torch
    from gwdetector.cnn import ChirpCNN, n_parameters, predict_scores
    from gwdetector.train import TrainConfig, train
    epochs, width = VARIANTS[variant]
    torch.manual_seed(seed)
    model = ChirpCNN(width)
    cfg = TrainConfig(epochs=epochs)
    t0 = time.time()
    history, best = train(model, cfg, SimConfig(), "cuda", run=seed, log=lambda m: None)
    model.load_state_dict(best)
    scores = predict_scores(model, X, "cuda")
    stem = RUNS / f"{variant}_s{seed}"
    np.save(f"{stem}_scores.npy", scores)
    b = min(history, key=lambda h: h["val_loss"])
    info = dict(variant=variant, seed=seed, params=n_parameters(model), epochs=epochs,
                best_epoch=b["epoch"], val_loss=b["val_loss"], val_auc=b["val_auc"],
                seconds=time.time() - t0)
    (RUNS / f"{variant}_s{seed}.json").write_text(json.dumps(dict(info, history=history)))
    log(json.dumps(info))


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    X = load_test_set()[0]
    with open(RUNS / "log.txt", "a") as f:
        def log(msg):
            f.write(msg + "\n"); f.flush()
        log(f"# start {time.strftime('%Y-%m-%d %H:%M:%S')} device={torch.cuda.get_device_name(0)}")
        for variant in args.variants:
            for seed in args.seeds:
                if (RUNS / f"{variant}_s{seed}_scores.npy").exists():
                    continue
                run_one(variant, seed, X, log)
        log("# all done")


if __name__ == "__main__":
    main()

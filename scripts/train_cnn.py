"""Train the CNN on the GPU. Saves the best (lowest validation loss) weights and the history."""
import argparse
import json
import time

import torch

from gwdetector.cnn import ChirpCNN, n_parameters
from gwdetector.data import PROJECT_ROOT
from gwdetector.simulate import SimConfig
from gwdetector.train import TrainConfig, train

RESULTS = PROJECT_ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    ap.add_argument("--cpu", action="store_true", help="allow CPU (default: require the GPU)")
    args = ap.parse_args()
    if not torch.cuda.is_available() and not args.cpu:
        raise SystemExit("CUDA GPU not available (pass --cpu to train on the CPU)")
    device = "cpu" if args.cpu else "cuda"
    torch.manual_seed(0)
    cfg = TrainConfig(epochs=args.epochs)
    model = ChirpCNN()
    name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"device: {name}; {n_parameters(model):,} parameters; {cfg.epochs} epochs x {cfg.n_train} examples")
    t0 = time.time()
    history, best = train(model, cfg, SimConfig(), device,
                          log=lambda m: print(m) if int(m.split()[1]) % 5 == 0 else None)
    RESULTS.mkdir(exist_ok=True)
    torch.save(best, RESULTS / "cnn_best.pt")
    b = min(history, key=lambda h: h["val_loss"])
    (RESULTS / "cnn_history.json").write_text(json.dumps(dict(
        device=name, train_seconds=time.time() - t0, config=cfg.__dict__, history=history), indent=2))
    print(f"done in {time.time() - t0:.0f}s; best epoch {b['epoch']} val loss {b['val_loss']:.4f} auc {b['val_auc']:.4f}")


if __name__ == "__main__":
    main()

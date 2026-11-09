"""Training loop for the CNN. Fresh simulated data every epoch, fixed validation set."""
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from gwdetector.cnn import predict_scores
from gwdetector.metrics import auc, roc_curve
from gwdetector.simulate import VAL_SEED, TRAIN_SEED, make_dataset


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 40
    n_train: int = 20000          # fresh examples per epoch (so the CNN never sees a repeat)
    n_val: int = 6000
    batch: int = 128
    lr: float = 1e-3
    snr_range: tuple = (1.0, 12.0)    # covers the whole 1-10 test range with a little margin
    workers: int = 6              # processes generating the next epoch's data (0 = inline)
    chunk_size: int = 500         # examples per work item (big pipe messages crash on Windows)


def train_seed(run, epoch, chunk=0):
    """Seed of one chunk of one epoch of training data for repeat number `run`.

    All values are ~1e11, far from VAL_SEED (2002) and the test-set seeds (3003..3013),
    so train/val/test never share noise or chirp parameters.
    """
    return TRAIN_SEED * 100_000_000 + run * 1_000_000 + epoch * 100 + chunk


def submit_epoch(pool, sim_cfg, cfg, run, epoch):
    """Start generating one epoch of data in chunks of cfg.chunk_size; returns the futures."""
    n_chunks = -(-cfg.n_train // cfg.chunk_size)                 # ceil
    assert n_chunks < 100, "train_seed reserves 100 chunk numbers per epoch"
    sizes = [min(cfg.chunk_size, cfg.n_train - i * cfg.chunk_size) for i in range(n_chunks)]
    return [pool.submit(make_dataset, sim_cfg, sizes[i], train_seed(run, epoch, i), cfg.snr_range)
            for i in range(n_chunks)]


def collect_epoch(futures):
    parts = [f.result() for f in futures]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def epoch_data(sim_cfg, cfg, run, epoch):
    """Same data as the pooled path, computed inline (used by tests)."""
    class Inline:
        def submit(self, fn, *a):
            from concurrent.futures import Future
            f = Future(); f.set_result(fn(*a)); return f
    return collect_epoch(submit_epoch(Inline(), sim_cfg, cfg, run, epoch))


def run_epoch(model, X, y, opt, loss_fn, batch, device):
    model.train()
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y.astype(np.float32)).to(device)
    perm = torch.randperm(len(Xt), device=device)
    total = 0.0
    for s in range(0, len(Xt), batch):
        idx = perm[s:s + batch]
        opt.zero_grad()
        loss = loss_fn(model(Xt[idx]), yt[idx])
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    return total / len(Xt)


def validate(model, Xv, yv, loss_fn, device):
    scores = predict_scores(model, Xv, device)
    loss = loss_fn(torch.from_numpy(scores), torch.from_numpy(yv.astype(np.float32))).item()
    return loss, auc(*roc_curve(scores[yv == 0], scores[yv == 1]))


def train(model, cfg, sim_cfg, device, run=0, log=print):
    """Train; returns (history, best_state_dict) where best = lowest validation loss.

    `run` selects the training-data seeds (repeat number). Data for epoch e+1 is generated
    by worker processes while the GPU trains on epoch e.
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    Xv, yv, _ = make_dataset(sim_cfg, cfg.n_val, VAL_SEED, cfg.snr_range)
    history, best, best_state = [], np.inf, None
    pool = ProcessPoolExecutor(cfg.workers) if cfg.workers > 0 else None
    try:
        pending = submit_epoch(pool, sim_cfg, cfg, run, 0) if pool else None
        for ep in range(cfg.epochs):
            t0 = time.time()
            if pool:
                X, y = collect_epoch(pending)
                if ep + 1 < cfg.epochs:
                    pending = submit_epoch(pool, sim_cfg, cfg, run, ep + 1)
            else:
                X, y = epoch_data(sim_cfg, cfg, run, ep)
            tr = run_epoch(model, X, y, opt, loss_fn, cfg.batch, device)
            vl, va = validate(model, Xv, yv, loss_fn, device)
            sched.step()
            history.append(dict(epoch=ep, train_loss=tr, val_loss=vl, val_auc=va, secs=time.time() - t0))
            if vl < best:
                best, best_state = vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            log(f"epoch {ep:2d} train {tr:.4f} val {vl:.4f} auc {va:.4f} ({time.time() - t0:.0f}s)")
    finally:
        if pool:
            pool.shutdown(wait=False, cancel_futures=True)
    return history, best_state

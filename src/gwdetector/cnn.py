"""A small 1D CNN that scores a 4 s strain segment: higher logit = more likely to hold a chirp."""
import numpy as np
import torch
from torch import nn


def conv_block(c_in, c_out, kernel, pool):
    """Conv -> batch-norm -> ReLU -> max-pool. 'same' padding keeps length; pooling shrinks it."""
    layers = [nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2),
              nn.BatchNorm1d(c_out), nn.ReLU()]
    if pool > 1:
        layers.append(nn.MaxPool1d(pool))
    return nn.Sequential(*layers)


class ChirpCNN(nn.Module):
    """(batch, n_samples) -> (batch,) logits.

    Five conv blocks shrink 8192 samples to 32 steps while channels grow 16 -> 128.
    `width` multiplies every channel count (width=2 is the "wider" network).
    A global max over time then makes the score independent of WHERE the chirp is.
    """

    def __init__(self, width=1):
        super().__init__()
        c = [16 * width, 32 * width, 64 * width, 128 * width, 128 * width]
        self.features = nn.Sequential(
            conv_block(1, c[0], 31, 4),       # 8192 -> 2048
            conv_block(c[0], c[1], 15, 4),    # -> 512
            conv_block(c[1], c[2], 7, 4),     # -> 128
            conv_block(c[2], c[3], 5, 2),     # -> 64
            conv_block(c[3], c[4], 5, 2),     # -> 32
        )
        self.head = nn.Sequential(nn.Linear(c[4], 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        h = self.features(x.unsqueeze(1))          # (batch, 128, steps)
        h = h.max(dim=2).values                    # global max pool over time
        return self.head(h).squeeze(1)


def n_parameters(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def predict_scores(model, X, device, batch=500):
    """Logits for every row of X (array or memmap), computed in batches."""
    model.eval()
    out = []
    for s in range(0, len(X), batch):
        xb = torch.from_numpy(np.array(X[s:s + batch], dtype=np.float32)).to(device)   # copy: memmaps are read-only
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)

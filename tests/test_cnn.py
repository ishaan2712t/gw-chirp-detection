"""CNN shape, shift-invariance, GPU use, and a short training run that must actually learn."""
import numpy as np
import pytest
import torch

from gwdetector.cnn import ChirpCNN, n_parameters, predict_scores
from gwdetector.simulate import SimConfig, TEST_SEED, VAL_SEED, make_dataset
from gwdetector.train import (TrainConfig, collect_epoch, epoch_data, submit_epoch, train,
                              train_seed)

CFG = SimConfig()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_output_shape_and_size():
    model = ChirpCNN()
    out = model(torch.zeros(3, CFG.n))
    assert out.shape == (3,)
    assert 50_000 < n_parameters(model) < 500_000          # "small"


def test_circular_shift_changes_score_only_slightly():
    torch.manual_seed(0)
    model = ChirpCNN().eval()
    x = torch.randn(4, CFG.n)
    shifted = torch.roll(x, 1024, dims=1)
    # global max pooling: shifting the input by whole pooling steps should barely matter
    assert torch.allclose(model(x), model(shifted), atol=0.5)


def test_predict_scores_batching_is_consistent():
    torch.manual_seed(0)
    model = ChirpCNN().to(DEVICE)
    X = np.random.default_rng(0).normal(size=(7, CFG.n)).astype(np.float32)
    np.testing.assert_allclose(predict_scores(model, X, DEVICE, batch=3),
                               predict_scores(model, X, DEVICE, batch=7), atol=1e-4)


def test_train_seeds_never_collide_with_val_or_test():
    seeds = {train_seed(r, e, c) for r in range(5) for e in range(130) for c in range(16)}
    assert len(seeds) == 5 * 130 * 16                         # all distinct
    assert VAL_SEED not in seeds
    assert not seeds & {TEST_SEED + k for k in range(11)}


def test_wider_network_has_about_four_times_the_parameters():
    ratio = n_parameters(ChirpCNN(width=2)) / n_parameters(ChirpCNN())
    assert 3 < ratio < 4.2
    assert ChirpCNN(width=2)(torch.zeros(2, CFG.n)).shape == (2,)


def test_parallel_data_generation_matches_inline():
    from concurrent.futures import ProcessPoolExecutor
    cfg = TrainConfig(n_train=60, workers=3, chunk_size=25)   # 3 chunks: 25+25+10
    Xi, yi = epoch_data(CFG, cfg, run=1, epoch=2)
    with ProcessPoolExecutor(3) as pool:
        Xp, yp = collect_epoch(submit_epoch(pool, CFG, cfg, 1, 2))
    np.testing.assert_array_equal(Xi, Xp)
    np.testing.assert_array_equal(yi, yp)
    assert Xi.shape == (60, CFG.n)
    Xo, _ = epoch_data(CFG, cfg, run=2, epoch=2)               # other repeat -> other data
    assert not np.allclose(Xi, Xo)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_model_runs_on_gpu():
    model = ChirpCNN().to("cuda")
    assert next(model.parameters()).is_cuda
    assert model(torch.zeros(2, CFG.n, device="cuda")).is_cuda


def test_short_training_learns_loud_signals():
    """At SNR 12-20 the problem is easy: validation AUC must clearly beat chance after 3 epochs."""
    torch.manual_seed(0)
    cfg = TrainConfig(epochs=3, n_train=1500, n_val=600, batch=64, snr_range=(12.0, 20.0), workers=2)
    history, state = train(ChirpCNN(), cfg, CFG, DEVICE, log=lambda *_: None)
    assert history[-1]["val_auc"] > 0.9
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert set(state) == set(ChirpCNN().state_dict())

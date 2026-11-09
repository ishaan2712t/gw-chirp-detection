"""Template bank over the (m1, m2) mass range, and a measurement of its mismatch."""
import numpy as np

from gwdetector.baseline import bank_statistics, template_matrix
from gwdetector.simulate import draw_parameters, unit_chirp
from gwdetector.template import chirp_mass, masses_from_mchirp_eta


def build_bank(cfg, mc_step=0.02, n_eta=4, margin=2.0):   # defaults: 102 templates, chosen by measuring mismatch
    """Grid in (chirp mass, eta): chirp mass is geometric (fractional step mc_step), eta linear.

    Keeps points whose masses lie inside the mass range (plus a small margin so that
    injections near the edge still have a neighbouring template). Returns (B, 2) of (m1, m2).
    """
    lo, hi = cfg.mass_range
    mc_lo, mc_hi = chirp_mass(lo, lo), chirp_mass(hi, hi)
    eta_lo = lo * hi / (lo + hi) ** 2
    mcs = mc_lo * (1 + mc_step) ** np.arange(0, int(np.log(mc_hi / mc_lo * 1.05) / np.log(1 + mc_step)) + 2)
    etas = np.linspace(eta_lo, 0.25, n_eta)
    params = []
    for mc in mcs:
        for eta in etas:
            m1, m2 = masses_from_mchirp_eta(mc, eta)
            if lo - margin <= m2 and m1 <= hi + margin:
                params.append((m1, m2))
    return np.array(params)


def measure_mismatch(cfg, params, n=300, seed=99):
    """Match = (largest noise-free SNR over the bank) / (injected SNR of 1).

    Random injections inside the mass range, never an exact bank template.
    mismatch = 1 - match; the SNR a bank template loses is (1 - match) of the signal.
    """
    m1, m2, tm = draw_parameters(cfg, np.random.default_rng(seed), n)
    X = np.stack([unit_chirp(cfg, a, b, t) for a, b, t in zip(m1, m2, tm)]).astype(np.float32)
    match, _ = bank_statistics(X, template_matrix(cfg, params), cfg, [(0.5, cfg.duration - 0.5)])
    return match[:, 0]

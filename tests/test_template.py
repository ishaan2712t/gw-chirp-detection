"""Check the chirp template against textbook formulas."""
import numpy as np

from gwdetector.template import (T_SUN, chirp_mass, isco_frequency, masses_from_mchirp_eta,
                                 taylorf2)

FS, N = 4096.0, 4096 * 32
FREQS = np.fft.rfftfreq(N, 1 / FS)


def test_mass_conversion_round_trip():
    m1, m2 = 36.0, 29.0
    eta = m1 * m2 / (m1 + m2) ** 2
    r1, r2 = masses_from_mchirp_eta(chirp_mass(m1, m2), eta)
    assert np.allclose([r1, r2], [m1, m2])


def test_support_is_flo_to_isco():
    h = taylorf2(FREQS, 30, 30, f_low=20, cut_factor=1.0)
    nz = FREQS[np.abs(h) > 0]
    assert nz.min() >= 20 and nz.max() <= isco_frequency(30, 30)
    assert abs(isco_frequency(30, 30) - 73.2) < 0.5     # 4400/60 ~ 73 Hz


def test_amplitude_scales_as_f_to_minus_7_6():
    h = taylorf2(FREQS, 30, 30)
    i, j = np.searchsorted(FREQS, [30, 60])
    ratio = np.abs(h[j]) / np.abs(h[i])
    assert abs(ratio / 2 ** (-7 / 6) - 1) < 1e-6


def test_time_before_merger_matches_newtonian_formula():
    """Group delay of h~(f) must be -tau(f), tau = 5/256 * Mc^(-5/3) * (pi f)^(-8/3)."""
    m1 = m2 = 30.0
    h = taylorf2(FREQS, m1, m2, pn_order=0)
    phase = np.unwrap(np.angle(h[np.abs(h) > 0]))
    f = FREQS[np.abs(h) > 0]
    t = -np.gradient(phase, f) / (2 * np.pi)          # arrival time = -(phase slope)/2pi
    mc = chirp_mass(m1, m2) * T_SUN
    for f0 in (25.0, 40.0, 60.0):
        k = np.searchsorted(f, f0)
        expected = -5 / 256 * mc ** (-5 / 3) * (np.pi * f0) ** (-8 / 3)
        assert abs(t[k] / expected - 1) < 0.01


def test_higher_pn_order_changes_phase_but_not_amplitude():
    h0 = taylorf2(FREQS, 36, 29, pn_order=0)
    h2 = taylorf2(FREQS, 36, 29, pn_order=2)
    np.testing.assert_allclose(np.abs(h0), np.abs(h2))
    assert np.max(np.abs(np.angle(h0 / np.where(h2 == 0, 1, h2)))) > 0.1


def test_real_time_domain_template_is_a_finite_chirp():
    h = np.fft.irfft(taylorf2(FREQS, 30, 30), n=N)
    assert np.isfinite(h).all() and np.abs(h).max() > 0


def test_cut_factor_extends_frequency_range():
    nz = lambda h: FREQS[np.abs(h) > 0].max()
    assert nz(taylorf2(FREQS, 30, 30, cut_factor=2.0)) > 1.9 * nz(taylorf2(FREQS, 30, 30, cut_factor=1.0))


def test_default_cut_is_three_times_isco():
    h = taylorf2(FREQS, 30, 30)
    assert FREQS[np.abs(h) > 0].max() <= 3 * isco_frequency(30, 30)
    assert FREQS[np.abs(h) > 0].max() > 2.9 * isco_frequency(30, 30)

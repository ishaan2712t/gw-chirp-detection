"""Inspiral chirp templates in the frequency domain (TaylorF2 / stationary phase)."""
import numpy as np

# Fixed at 3x (picked after a sweep on GW150914; not retuned per event): cut the template at 3x the ISCO frequency, a crude
# stand-in for the merger, which the pure inspiral formula cannot describe.
DEFAULT_CUT_FACTOR = 3.0
T_SUN = 4.925491e-6        # G*Msun/c^3 in seconds: converts solar masses to seconds


def chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def masses_from_mchirp_eta(mchirp, eta):
    """Invert (Mc, eta) -> (m1, m2) with m1 >= m2."""
    total = mchirp * eta ** (-0.6)
    root = np.sqrt(1 - 4 * eta)
    return total * (1 + root) / 2, total * (1 - root) / 2


def isco_frequency(m1, m2):
    """Frequency where the inspiral template stops (innermost stable circular orbit)."""
    return 1.0 / (6**1.5 * np.pi * (m1 + m2) * T_SUN)


def pn_phase(f, m1, m2, pn_order=2):
    """Orbital phase Psi(f) [rad] without the arbitrary tc, phi_c terms.

    pn_order=0 is the leading-order (Newtonian) chirp; 2 adds the 1PN, 1.5PN and
    2PN corrections (non-spinning).
    """
    total = (m1 + m2) * T_SUN
    eta = m1 * m2 / (m1 + m2) ** 2
    v = (np.pi * total * f) ** (1 / 3)
    series = np.ones_like(v)
    if pn_order >= 2:
        series += (3715 / 756 + 55 * eta / 9) * v**2
        series += -16 * np.pi * v**3
        series += (15293365 / 508032 + 27145 * eta / 504 + 3085 * eta**2 / 72) * v**4
    return 3 / (128 * eta * v**5) * series


def taylorf2(freqs, m1, m2, f_low=20.0, pn_order=2, cut_factor=DEFAULT_CUT_FACTOR):
    """Template h~(f) on the given rfft frequency grid, merger placed at t = 0.

    Amplitude ~ f^(-7/6) (overall scale is irrelevant: SNR is normalised).
    Zero below f_low and above cut_factor * f_ISCO. The inspiral formula is only
    trustworthy up to ~f_ISCO; larger factors crudely stand in for the merger.
    Sign convention matches numpy's FFT (e^{-2 pi i f t}).
    """
    f_high = cut_factor * isco_frequency(m1, m2)
    h = np.zeros(freqs.shape, dtype=complex)
    ok = (freqs >= f_low) & (freqs <= f_high)
    f = freqs[ok]
    h[ok] = f ** (-7 / 6) * np.exp(-1j * (pn_phase(f, m1, m2, pn_order) - np.pi / 4))
    return h

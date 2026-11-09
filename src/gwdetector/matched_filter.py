"""FFT matched filter: slide a (whitened) template along (whitened) data."""
import numpy as np


def template_norm(Hw, n):
    """sqrt(sum of h_w[t]^2), computed from the rfft spectrum (Parseval).

    Dividing by this makes pure-noise output have unit variance in each of the
    real/imaginary parts, so the output is directly an SNR.
    """
    power = np.abs(Hw) ** 2
    power[1:] *= 2                       # negative frequencies mirror the positive ones
    if n % 2 == 0:
        power[-1] /= 2                   # Nyquist bin has no mirror
    return np.sqrt(power.sum() / n)


def snr_series(Xw, Hw, n):
    """SNR(t) for every time shift t, maximised over the template's phase.

    Xw, Hw: whitened rfft spectra of data and template (same length n).
    Correlation = IFFT(X * conj(H)). Keeping only positive frequencies (and doubling)
    gives a complex series whose real part is the correlation with the template and
    whose imaginary part is the correlation with the 90-degree-shifted template;
    |z| is therefore independent of the unknown orbital phase.
    """
    Z = np.zeros(n, dtype=complex)
    k = len(Xw)
    Z[:k] = Xw * np.conj(Hw)
    Z[0] = 0                             # DC is band-passed away anyway
    if n % 2 == 0:
        Z[k - 1] = 0                     # Nyquist has no positive-frequency partner
    z = 2 * np.fft.ifft(Z)
    return np.abs(z) / template_norm(Hw, n)


def find_peak(snr, fs, edge=2.0):
    """(value, index) of the largest SNR, ignoring `edge` seconds at both ends
    (where the taper and filter wrap-around corrupt the data)."""
    e = int(edge * fs)
    i = e + int(np.argmax(snr[e:-e]))
    return float(snr[i]), i

"""Tier 1: re-detect GW150914 in H1 and L1 with a chirp-mass scan of matched filters."""
import json

import matplotlib.pyplot as plt
import numpy as np
from gwosc.datasets import event_gps
from scipy.ndimage import maximum_filter1d

from gwdetector.data import DETECTORS, FIG_DIR, PROJECT_ROOT, fetch_strain
from gwdetector.matched_filter import find_peak, snr_series
from gwdetector.psd import welch_psd
from gwdetector.template import masses_from_mchirp_eta, taylorf2
from gwdetector.whiten import whiten, whiten_fd, whitened_spectrum

EVENT = "GW150914"
ETA = 0.8 / 1.8**2            # fixed mass ratio q = 0.8 while scanning the chirp mass
MCHIRPS = np.arange(20.0, 40.1, 1.0)
COINC = 0.010                 # seconds: allowed H1-L1 arrival-time difference
EDGE = 2.0


def prepare(ifo):
    """Strain -> PSD -> whitened data spectrum for one detector."""
    ts = fetch_strain(EVENT, ifo, 32)
    x, fs = ts.value, ts.sample_rate.value
    pf, psd, _ = welch_psd(x, fs, 4.0, 0.5, average="median")
    return dict(x=x, fs=fs, t0=ts.t0.value, pf=pf, psd=psd,
                Xw=whitened_spectrum(x, fs, pf, psd))


def snr_for(det, mchirp, pn, cut):
    n, fs = len(det["x"]), det["fs"]
    f = np.fft.rfftfreq(n, 1 / fs)
    m1, m2 = masses_from_mchirp_eta(mchirp, ETA)
    Hw = whiten_fd(taylorf2(f, m1, m2, pn_order=pn, cut_factor=cut), f, det["pf"], det["psd"], fs)
    return snr_series(det["Xw"], Hw, n)


def best_coincident(dets, pn, cut):
    """For each template, maximise sqrt(H1(t)^2 + max_{|dt|<10ms} L1(t+dt)^2) over t."""
    best = None
    for mc in MCHIRPS:
        h, l = (snr_for(dets[i], mc, pn, cut) for i in ("H1", "L1"))
        win = int(2 * COINC * dets["H1"]["fs"]) + 1
        l_near = maximum_filter1d(l, size=win)
        comb = np.sqrt(h**2 + l_near**2)
        e = int(EDGE * dets["H1"]["fs"])
        i = e + int(np.argmax(comb[e:-e]))
        if best is None or comb[i] > best["combined"]:
            best = dict(mchirp=float(mc), combined=float(comb[i]), idx=i, h=h, l=l)
    return best


def summarise(dets, best):
    fs = dets["H1"]["fs"]
    i = best["idx"]
    w = int(COINC * fs)
    j = i - w + int(np.argmax(best["l"][i - w:i + w + 1]))
    return dict(mchirp=best["mchirp"], H1_snr=float(best["h"][i]), L1_snr=float(best["l"][j]),
                combined=best["combined"], dt_L1_minus_H1_ms=float((j - i) / fs * 1e3),
                gps_H1=float(dets["H1"]["t0"] + i / fs)), i, j


def plot(dets, best, i, j):
    gps = event_gps(EVENT)
    fs = dets["H1"]["fs"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for col, ifo in enumerate(("H1", "L1")):
        d = dets[ifo]
        t = d["t0"] + np.arange(len(d["x"])) / fs - gps
        xw = whiten(d["x"], fs, psd_freqs=d["pf"], psd=d["psd"])
        m = (t > -0.4) & (t < 0.2)
        ax[0, col].plot(t[m], xw[m], lw=0.8)
        ax[0, col].set_title(f"{DETECTORS[ifo]}: whitened, 20–500 Hz")
        ax[0, col].set_xlabel("time from event [s]")
        snr = best["h" if ifo == "H1" else "l"]
        ax[1, col].plot(t, snr, lw=0.6)
        ax[1, col].set_xlim(-2, 2)
        ax[1, col].set_title(f"{ifo} matched-filter SNR (Mc={best['mchirp']:.0f})")
        ax[1, col].set_xlabel("time from event [s]")
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "04_gw150914_detection.png", dpi=130)


def main():
    dets = {ifo: prepare(ifo) for ifo in DETECTORS}
    out = {}
    for pn, cut in [(0, 1), (2, 1), (2, 2), (2, 3), (2, 4)]:
        best = best_coincident(dets, pn, cut)
        key = f"pn{pn}_cut{cut}"
        out[key], i, j = summarise(dets, best)
        r = out[key]
        print(f"PN{pn} cut={cut}xISCO: Mc={r['mchirp']:.0f} H1={r['H1_snr']:.1f} L1={r['L1_snr']:.1f} "
              f"combined={r['combined']:.1f} dt={r['dt_L1_minus_H1_ms']:+.1f} ms")
        if (pn, cut) == (2, 3):
            plot(dets, best, i, j)
            # noise-only SNR level away from the event, as a sanity check
            quiet = np.r_[best["h"][int(EDGE * dets['H1']['fs']):i - 4096], best["h"][i + 4096:-int(EDGE * dets['H1']['fs'])]]
            out["H1_noise_mean_snr"] = float(quiet.mean())
            print(f"H1 off-event mean SNR = {quiet.mean():.2f} (pure noise expects ~1.25)")
    path = PROJECT_ROOT / "results"
    path.mkdir(exist_ok=True)
    (path / "tier1_gw150914.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

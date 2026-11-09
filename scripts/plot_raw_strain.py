"""Download 32 s of H1 and L1 strain around GW150914 and plot it, unprocessed."""
import matplotlib.pyplot as plt
from gwosc.datasets import event_gps

from gwdetector.data import DETECTORS, FIG_DIR, fetch_strain

EVENT = "GW150914"
COLORS = {"H1": "#d62728", "L1": "#1f77b4"}


def main():
    gps = event_gps(EVENT)
    strain = {ifo: fetch_strain(EVENT, ifo, duration=32) for ifo in DETECTORS}

    for ifo, ts in strain.items():
        print(f"{ifo}: {len(ts)} samples @ {ts.sample_rate.value:.0f} Hz, "
              f"t0={ts.t0.value}, mean={ts.value.mean():.3e}, std={ts.value.std():.3e}")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    # Panels 1-2: full 32 s for each detector
    for ax, (ifo, ts) in zip(axes[:2], strain.items()):
        t = ts.times.value - gps
        ax.plot(t, ts.value, color=COLORS[ifo], lw=0.5)
        ax.axvline(0, color="k", ls="--", lw=0.8, label=f"{EVENT} merger time")
        ax.set_title(f"{DETECTORS[ifo]} ({ifo}) — raw strain")
        ax.set_ylabel("strain h(t)")
        ax.legend(loc="upper right")
    # Panel 3: zoom +/-0.2 s around the event, both detectors overlaid
    for ifo, ts in strain.items():
        z = ts.crop(gps - 0.2, gps + 0.2)
        axes[2].plot(z.times.value - gps, z.value, color=COLORS[ifo], label=ifo)
    axes[2].axvline(0, color="k", ls="--", lw=0.8)
    axes[2].set_title("Zoom: ±0.2 s around the merger (still raw — no signal visible)")
    axes[2].set_ylabel("strain h(t)")
    axes[2].legend(loc="upper right")
    axes[2].set_xlabel(f"time from GPS {gps} [s]")
    axes[1].set_xlabel(f"time from GPS {gps} [s]")

    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "01_raw_strain.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

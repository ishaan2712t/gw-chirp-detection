"""Stage 0: a small look at the data we will work with (GW150914 + 10 min of background)."""
from gwosc.datasets import event_gps
from gwosc.timeline import get_segments

from gwdetector.data import DATA_DIR, fetch_strain
from gwpy.timeseries import TimeSeries

DURATION = 600          # seconds of background to peek at
SEARCH = (1126400000, 1127000000)   # O1, ~1 week after GW150914; no catalogued events here


def overlap_segment(min_len):
    """First stretch where H1 and L1 are both in science mode for >= min_len seconds."""
    segs = {i: get_segments(f"{i}_DATA", *SEARCH) for i in ("H1", "L1")}
    for a0, a1 in segs["H1"]:
        for b0, b1 in segs["L1"]:
            s, e = max(a0, b0), min(a1, b1)
            if e - s >= min_len:
                return int(s), int(s) + min_len
    raise RuntimeError("no common segment found")


def fetch_background(ifo, start, end):
    path = DATA_DIR / "survey" / f"{ifo}_{start}_{end}.hdf5"
    path.parent.mkdir(exist_ok=True, parents=True)
    if path.exists():
        return TimeSeries.read(path), path
    ts = TimeSeries.fetch_open_data(ifo, start, end, sample_rate=4096)
    ts.write(path)
    return ts, path


def describe(label, ifo, ts, path):
    asd100 = ts.asd(fftlength=4, method="median").crop(99, 101).value.mean()
    print(f"{label:11s} {ifo}: {ts.duration.value:5.0f} s, {ts.sample_rate.value:.0f} Hz, "
          f"{path.stat().st_size / 1e6:5.1f} MB, nan={int(ts.value.size - (ts.value == ts.value).sum())}, "
          f"std={ts.value.std():.2e}, ASD@100Hz={asd100:.2e}")


def main():
    print(f"GW150914 GPS = {event_gps('GW150914')}")
    for ifo in ("H1", "L1"):
        ts = fetch_strain("GW150914", ifo, 32)
        describe("GW150914", ifo, ts, DATA_DIR / f"{ifo}_GW150914_32s_4096Hz.hdf5")
    start, end = overlap_segment(DURATION)
    print(f"background stretch: GPS {start}-{end} (both detectors in science mode)")
    for ifo in ("H1", "L1"):
        ts, path = fetch_background(ifo, start, end)
        describe("background", ifo, ts, path)


if __name__ == "__main__":
    main()

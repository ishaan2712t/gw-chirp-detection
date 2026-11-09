"""Download and cache GWOSC strain data."""
from pathlib import Path

from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"

DETECTORS = {"H1": "LIGO Hanford", "L1": "LIGO Livingston"}


def fetch_strain(event="GW150914", ifo="H1", duration=32, sample_rate=4096):
    """Return `duration` seconds of strain centred on `event` for detector `ifo`.

    The first call downloads from GWOSC; later calls read a local HDF5 copy.
    """
    gps = event_gps(event)
    start, end = gps - duration / 2, gps + duration / 2
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"{ifo}_{event}_{duration}s_{sample_rate}Hz.hdf5"
    if path.exists():
        return TimeSeries.read(path)
    print(f"Downloading {ifo} {event} [{start}, {end}) from GWOSC ...")
    ts = TimeSeries.fetch_open_data(ifo, start, end, sample_rate=sample_rate, verbose=False)
    ts.write(path)
    return ts

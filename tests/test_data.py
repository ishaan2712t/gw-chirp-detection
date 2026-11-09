"""fetch_strain: downloads once, then reads the local cache (no network in tests)."""
import numpy as np
from gwpy.timeseries import TimeSeries

from gwdetector import data


def _fake_series(start=984.0):
    return TimeSeries(np.arange(8, dtype=float), t0=start, sample_rate=4, name="X:STRAIN")


def test_cache_miss_downloads_and_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data, "event_gps", lambda event: 1000.0)
    calls = []

    def fake_fetch(ifo, start, end, **kw):
        calls.append((ifo, start, end))
        return _fake_series(start)

    monkeypatch.setattr(TimeSeries, "fetch_open_data", staticmethod(fake_fetch))
    ts = data.fetch_strain("EV", "H1", duration=32, sample_rate=4)
    assert calls == [("H1", 984.0, 1016.0)]       # window is centred on the event
    assert len(ts) == 8
    assert (tmp_path / "H1_EV_32s_4Hz.hdf5").exists()


def test_cache_hit_does_not_touch_network(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data, "event_gps", lambda event: 1000.0)
    _fake_series().write(tmp_path / "L1_EV_32s_4Hz.hdf5")

    def boom(*a, **k):
        raise AssertionError("network used despite cache")

    monkeypatch.setattr(TimeSeries, "fetch_open_data", staticmethod(boom))
    ts = data.fetch_strain("EV", "L1", duration=32, sample_rate=4)
    np.testing.assert_allclose(ts.value, np.arange(8))

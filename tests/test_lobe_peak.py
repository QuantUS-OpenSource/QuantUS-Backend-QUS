"""find_bw_edges(center_freq=...): with several lobes, the peak is in the lobe closest to the probe's
centre frequency instead of the global maximum (e.g. an L20 spectrum split by a dip where the lower
lobe is louder). Without center_freq the behaviour is unchanged (FindBWEdges.m)."""

import types

import numpy as np

from quantus.seg_loading.correlation_length.first_pass import find_bw_edges, probe_center_frequency_mhz
from quantus.seg_loading.power_spectrum_mtm.bandwidth import bw_options

FREQ = np.linspace(0, 30, 301)          # MHz, 0.1 MHz bins


def two_lobes(low_db=60.0, high_db=55.0, dip_db=20.0):
    """Lower lobe at 5 MHz (louder by default), upper lobe at 11 MHz, a deep dip at 7.5 MHz, a flat
    noise floor of 20 dB above ~15 MHz. Linear power."""
    s = np.full(FREQ.size, 20.0)
    s = np.maximum(s, low_db - ((FREQ - 5.0) / 1.2) ** 2)
    s = np.maximum(s, high_db - ((FREQ - 11.0) / 1.6) ** 2)
    s[np.abs(FREQ - 7.5) < 0.6] = dip_db
    return 10 ** (s / 10)


def test_without_center_freq_global_maximum_wins():
    fl, fh, info = find_bw_edges(two_lobes(), FREQ)
    assert info["peak_rule"] == "global maximum"
    assert abs(info["peak_freq"] - 5.0) < 0.15
    assert fh < 7.5                                   # band = the lower lobe only


def test_center_freq_picks_the_closer_lobe():
    fl, fh, info = find_bw_edges(two_lobes(), FREQ, center_freq=14.0)
    assert info["peak_rule"].startswith("lobe closest to 14")
    assert abs(info["peak_freq"] - 11.0) < 0.15
    assert fl > 7.5 and fh > 11.0                     # band = the upper lobe
    assert len(info["lobes_mhz"]) == 2


def test_center_freq_near_the_louder_lobe_keeps_it():
    fl, fh, info = find_bw_edges(two_lobes(), FREQ, center_freq=4.0)
    assert abs(info["peak_freq"] - 5.0) < 0.15 and fh < 7.5


def test_single_lobe_unchanged_by_center_freq():
    one = two_lobes(dip_db=58.0)                       # dip filled in: one lobe
    a = find_bw_edges(one, FREQ)
    b = find_bw_edges(one, FREQ, center_freq=14.0)
    assert a[:2] == b[:2] and a[2]["peak_freq"] == b[2]["peak_freq"]


def test_mtm_options_pass_center_freq_with_no_dip_detection():
    opts = bw_options((), 0.3, center_freq=14.0)
    assert opts["center_freq"] == 14.0 and np.isinf(opts["dip_prominence"])
    # the MTM band (no dip detection): lobes are split where the spectrum drops below threshold
    fl, fh, info = find_bw_edges(two_lobes(), FREQ, **opts)
    assert abs(info["peak_freq"] - 11.0) < 0.15


def test_probe_center_frequency_from_rf_yml(tmp_path):
    raw = tmp_path / "scan_rf.raw"
    raw.write_bytes(b"")
    (tmp_path / "scan_rf.yml").write_text("probe:\n  pitch: 130.00\ntransmit frequency: 14.00 MHz\n")
    assert probe_center_frequency_mhz(types.SimpleNamespace(scan_path=str(raw))) == 14.0
    assert probe_center_frequency_mhz(types.SimpleNamespace(center_frequency=3.5e6)) == 3.5
    assert probe_center_frequency_mhz(types.SimpleNamespace()) is None


def test_valid_band_rejects_reversed_and_missing():
    from quantus.seg_loading.correlation_length.roi import valid_band
    assert valid_band((8.62, 12.93))
    assert not valid_band((8.95, 0.99))           # reversed: no shared band, saved anyway
    assert not valid_band((np.nan, 12.0)) and not valid_band(None) and not valid_band((5.0, 5.0))

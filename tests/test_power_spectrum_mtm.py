"""Multitaper settings (CheckMTMParameters / DetermineSpectralParameters rules) and the ROI band."""

import numpy as np
import pytest

from quantus.seg_loading.power_spectrum_mtm import (
    bw_options, check_mtm_parameters, determine_spectral_parameters, dpss_tapers, matlab_prctile,
    power_spectrum_mtf, roi_bandwidth, round_band_inward, window_samples,
)


def test_band_rounds_inward_without_float_bumps():
    assert round_band_inward(1.538, 4.487) == (1.6, 4.4)
    assert round_band_inward(1.7, 3.6) == (1.7, 3.6)
    assert round_band_inward(1.7000000001, 3.5999999999) == (1.7, 3.6)


def test_window_samples_is_odd():
    assert window_samples(5.6, 5.6 / 109.2) == 109
    assert window_samples(5.6, 5.6 / 110) == 111


def test_check_mtm_parameters_values():
    c = check_mtm_parameters(109, (1.6, 4.4), 15.0, kappa=0.2)
    assert c.t_us == pytest.approx(109 / 15)
    assert c.nw_recommended == pytest.approx(0.2 * 2.8 * 109 / 15)
    np.testing.assert_array_equal(c.k, [3, 5, 7])
    np.testing.assert_allclose(c.w_mhz, np.array([2, 3, 4]) * 15 / 109)
    assert c.status == ["OK", "OK", "OK"]
    assert np.isnan(c.nw_max_before_leak)
    assert c.n_for_nw2 == int(np.ceil(2 * 15 / (0.2 * 2.8)))


def test_nw_rules_round_cap_and_flag():
    axial = np.repeat(np.arange(400)[:, None] * 0.05, 5, axis=1)
    p = determine_spectral_parameters((1.538, 4.487), 5.6, axial, 15.0)      # 5.6 / 0.05 = 112 -> 113
    assert p.n == 113 and p.band_mhz == (1.6, 4.4)
    assert p.nw == np.round(0.2 * 2.8 * 113 / 15 * 2) / 2 and p.k == int(2 * p.nw - 1)
    assert p.w_mhz == pytest.approx(p.nw * 15 / 113) and p.note == ""
    # a null 0.3 MHz below the band caps W at 0.3 MHz -> NW floor(0.3 T x 2) / 2
    q = determine_spectral_parameters((1.6, 4.4), 5.6, axial, 15.0, avoid_freq=[1.3])
    assert q.nw == np.floor(0.3 * 113 / 15 * 2) / 2 and q.note.startswith("capped")
    # narrow band -> NW < 1.5 flagged
    r = determine_spectral_parameters((2.0, 2.3), 2.0, axial, 15.0)
    assert r.nw < 1.5 and "single taper" in r.note
    with pytest.raises(ValueError):
        determine_spectral_parameters((2.01, 2.05), 5.6, axial, 15.0)


def test_matlab_prctile():
    x = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    assert matlab_prctile(x, 50) == 2.5
    assert matlab_prctile(x, 95) == 4.0          # above the last midpoint -> max
    assert matlab_prctile(x, 5) == 1.0
    assert matlab_prctile(x, 60) == pytest.approx(2.9)   # 4 * 0.6 + 0.5 = 2.9th value


def test_roi_bandwidth_shared_band_and_usable_windows():
    fs, n = 15.0, 101
    rng = np.random.default_rng(1)
    tapers = dpss_tapers(n, 3)
    t = np.arange(n) / fs

    def spectrum(f_lo, f_hi):
        sig = sum(np.cos(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f in np.arange(f_lo, f_hi, 0.05))
        return power_spectrum_mtf((sig + 1e-3 * rng.normal(size=n))[:, None], tapers, fs)[:, 0]

    freq = np.arange(513) * fs / 1024
    ref = np.stack([np.stack([spectrum(1.5, 4.5) for _ in range(2)]) for _ in range(2)])
    smp = ref.copy()
    smp[0, 0] = spectrum(2.0, 3.5)                    # narrower sample band in one window
    smp[1, 1] = 1e-12 * np.ones(513)                  # no signal: no band
    b = roi_bandwidth(smp, ref, freq, **bw_options(w_mhz=0.4))
    assert b.n_windows == 4 and b.n_usable == 3 and not b.shared_usable[1, 1]
    assert b.shared_low[0, 0] == max(b.sample_low[0, 0], b.ref_low[0, 0])
    assert b.shared_high[0, 0] == min(b.sample_high[0, 0], b.ref_high[0, 0])
    u = b.shared_usable
    assert b.shared_strict[0] == np.ceil(np.nanmax(b.shared_low[u]) * 10 - 1e-6) / 10
    assert b.shared_strict[1] == np.floor(np.nanmin(b.shared_high[u]) * 10 + 1e-6) / 10
    m = np.array([[True, False], [False, False]])
    bm = roi_bandwidth(smp, ref, freq, mask=m, **bw_options(w_mhz=0.4))
    assert bm.n_windows == 1 and np.isnan(bm.sample_low[1, 0])


def test_bw_options_match_determine_bandwidth():
    # center_freq None = global maximum, as DetermineBandwidth.m
    assert bw_options() == dict(avoid_freq=(), search_above=-np.inf, guard=0.0, dip_prominence=np.inf,
                                center_freq=None)
    assert bw_options([7.5], 0.3)["search_above"] == 7.5
    assert bw_options([7.5], 0.3, center_freq=14.0)["center_freq"] == 14.0

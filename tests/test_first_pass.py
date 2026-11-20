"""First-pass power check (port of CalcFirstPassPowerSpectra / FindBWEdges / IntersectFirstPass /
ROIBandCheck). The port was checked field by field against the MATLAB on the same RFData.mat
frames (see quantus/seg_loading/correlation_length/README_first_pass.md); these tests pin the behaviour on
synthetic data."""

import warnings

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from quantus.seg_loading.correlation_length.first_pass import (
    NyquistError, NyquistWarning, calc_first_pass, find_bw_edges, inpolygon, intersect_first_pass, matlab_hanning,
    roi_band_check,
)


def test_matlab_hanning_has_no_zero_end_points():
    np.testing.assert_allclose(matlab_hanning(3), [0.5, 1.0, 0.5])
    w = matlab_hanning(7)
    assert w[0] > 0 and np.isclose(w[3], 1.0) and np.allclose(w, w[::-1])


def _spectrum(freq, f0=5.0, width=1.5, floor_db=-40.0, peak_db=0.0):
    """Gaussian band on a flat noise floor, linear power."""
    sig = 10 ** (peak_db / 10) * np.exp(-0.5 * ((freq - f0) / width) ** 2)
    return sig + 10 ** (floor_db / 10)


def test_band_edges_are_noise_floor_plus_10_db():
    freq = np.linspace(0, 20, 401)
    fl, fh, info = find_bw_edges(_spectrum(freq, f0=8.0), freq)
    assert info["noise_method_used"] == "plateau"
    # plateau = bins within 3 dB of the high-side 20th percentile, so the band's tail lifts it slightly
    assert info["noise_floor_db"] == pytest.approx(-40, abs=0.3)
    assert info["threshold_db"] == pytest.approx(info["noise_floor_db"] + 10)
    assert info["signal_power_db"] == pytest.approx(-info["threshold_db"], abs=1e-3)  # peak ~0 dB (+ floor)
    # gaussian at ~-30 dB (re peak): f0 +/- width * sqrt(2 * 3 ln 10)
    half = 1.5 * np.sqrt(2 * 3 * np.log(10))
    assert fl == pytest.approx(8 - half, abs=0.1) and fh == pytest.approx(8 + half, abs=0.1)
    assert info["low_limited_by"] == info["high_limited_by"] == "threshold"


def test_walk_stops_at_a_notch_and_guard_keeps_clear():
    freq = np.linspace(0, 20, 401)
    ps = _spectrum(freq) * (1 - 0.97 * np.exp(-0.5 * ((freq - 3.5) / 0.15) ** 2))  # 15 dB notch at 3.5 MHz
    fl, fh, info = find_bw_edges(ps, freq)
    assert info["low_limited_by"].startswith("notch") and fl == pytest.approx(3.5, abs=0.06)
    fl_g, _, info_g = find_bw_edges(ps, freq, guard=0.5)
    assert fl_g == pytest.approx(fl + 0.5, abs=0.06) and "guard" in info_g["low_limited_by"]
    fl_a, _, _ = find_bw_edges(_spectrum(freq), freq, avoid_freq=(2.5,), guard=0.2)
    assert fl_a == pytest.approx(2.7)


def test_band_running_to_nyquist_and_no_band():
    freq = np.linspace(0, 10, 201)
    rising = 10 ** (np.linspace(-40, 0, 201) / 10)  # signal up to Nyquist: no high-side plateau
    fl, fh, info = find_bw_edges(rising, freq)
    assert info["noise_method_used"].startswith("legacy") and info["high_limited_by"] == "spectrum edge"
    assert fh == pytest.approx(freq[-2])  # last bin dropped (Clarius clipping)
    flat = np.ones(201)
    fl, fh, info = find_bw_edges(flat, freq)
    assert np.isnan(fl) and np.isnan(fh)  # no band: peak never clears the floor + 10 dB


def test_noise_band_method_and_search_above():
    freq = np.linspace(0, 20, 401)
    ps = _spectrum(freq) + 0.3 * np.exp(-0.5 * ((freq - 12) / 0.8) ** 2)  # weaker upper lobe
    _, _, info = find_bw_edges(ps, freq, noise_method="band", noise_band=(17, 19))
    assert info["noise_method_used"] == "band" and info["noise_floor_db"] == pytest.approx(-40, abs=0.1)
    _, _, info = find_bw_edges(ps, freq, search_above=9)
    assert info["peak_freq"] == pytest.approx(12, abs=0.06)
    with pytest.raises(ValueError):
        find_bw_edges(ps, freq, noise_method="band")


def test_inpolygon_counts_points_on_the_boundary():
    xv, yv = [0, 2, 2, 0], [0, 0, 1, 1]
    x = np.array([1.0, 0.0, 2.0, 1.0, 3.0, 1.0])
    y = np.array([0.5, 0.5, 1.0, 0.0, 0.5, 1.5])
    np.testing.assert_array_equal(inpolygon(x, y, xv, yv), [True, True, True, True, False, False])


def _band_limited_rf(rng, n_frames, n_samples, n_lines, fs, band, noise=0.02):
    sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
    rf = sosfiltfilt(sos, rng.normal(size=(n_frames, n_samples, n_lines)), axis=1)
    return rf / rf.std() + noise * rng.normal(size=rf.shape)


def _linear_positions(n_samples, n_lines, fs_mhz, pitch=0.2):
    d_ax = 1.540 / 2 / fs_mhz
    axial = np.repeat(np.arange(n_samples)[:, None] * d_ax, n_lines, axis=1)
    lateral = np.repeat(((np.arange(n_lines) - (n_lines - 1) / 2) * pitch)[None], n_samples, axis=0)
    return axial, lateral


def test_first_pass_finds_the_signal_band_and_maps_usable_blocks():
    rng = np.random.default_rng(0)
    fs, n_samples, n_lines = 40.0, 1200, 64
    rf = _band_limited_rf(rng, 3, n_samples, n_lines, fs, (4, 8))
    rf[:, :, 40:] = 0.02 * rng.normal(size=(3, n_samples, n_lines - 40))  # right side: noise only
    axial, lateral = _linear_positions(n_samples, n_lines, fs)
    fp = calc_first_pass(rf, axial, lateral, window_length_mm=2.0, block_width_mm=2.0, ax_step_mm=1.0)
    assert fp.fs_mhz == pytest.approx(fs)
    assert fp.window_samples % 2 == 1 and fp.freq_axis[1] == pytest.approx(fs / fp.window_samples)
    assert np.all(fp.usable)
    # 4-8 MHz pass band; edges at noise floor + 10 dB lie outside the -3 dB points
    assert 2.5 <= np.nanmedian(fp.f_low) <= 4.5 and 7.5 <= np.nanmedian(fp.f_high) <= 10.0
    signal_blocks = fp.block_centre + fp.block_lines // 2 < 40
    noise_blocks = fp.block_start >= 40
    assert fp.usable_2d[:, signal_blocks].all() and not fp.usable_2d[:, noise_blocks].any()
    # explicit sampling frequency rescales the frequency axis only
    fp2 = calc_first_pass(rf, axial, lateral, 2.0, 2.0, ax_step_mm=1.0, sampling_frequency_mhz=2 * fs)
    np.testing.assert_allclose(fp2.freq_axis, 2 * fp.freq_axis)
    np.testing.assert_allclose(fp2.mean_ps, fp.mean_ps)
    with pytest.raises(ValueError, match="ax_step_mm or n_ax_locations"):
        calc_first_pass(rf, axial, lateral, 2.0, 2.0)


def test_signal_up_to_nyquist_stops_the_first_pass():
    rng = np.random.default_rng(2)
    fs, n_samples, n_lines = 20.0, 800, 40
    axial, lateral = _linear_positions(n_samples, n_lines, fs)
    # like Clarius L15 at fs 20 MHz: band from ~5 MHz still well above the noise at Nyquist (10 MHz)
    rf = sosfiltfilt(butter(2, 5.0, btype="highpass", fs=fs, output="sos"), rng.normal(size=(2, n_samples, n_lines)),
                     axis=1)
    rf = sosfiltfilt(butter(1, 8.0, btype="lowpass", fs=fs, output="sos"), rf, axis=1)
    rf += 0.01 * rng.normal(size=rf.shape)
    kw = dict(window_length_mm=2.0, block_width_mm=2.0, ax_step_mm=1.0)
    with pytest.raises(NyquistError, match="Nyquist"):
        calc_first_pass(rf, axial, lateral, **kw)
    with pytest.warns(NyquistWarning):
        fp = calc_first_pass(rf, axial, lateral, on_nyquist="warn", **kw)
    assert fp.nyquist_limited.all()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert calc_first_pass(rf, axial, lateral, on_nyquist="ignore", **kw).nyquist_limited.all()
    # a band well inside Nyquist passes, with nothing flagged
    ok = _band_limited_rf(rng, 2, n_samples, n_lines, fs, (2, 5))
    assert not calc_first_pass(ok, axial, lateral, **kw).nyquist_limited.any()


def test_intersect_and_roi_band_of_a_given_roi():
    rng = np.random.default_rng(1)
    fs, n_samples, n_lines = 40.0, 1600, 80
    axial, lateral = _linear_positions(n_samples, n_lines, fs)
    ref = _band_limited_rf(rng, 3, n_samples, n_lines, fs, (3, 9))
    smp = _band_limited_rf(rng, 1, n_samples, n_lines, fs, (4, 8))
    smp[:, 1100:, :] = 0.02 * rng.normal(size=(1, n_samples - 1100, n_lines))  # sample loses signal deep
    kw = dict(window_length_mm=2.0, block_width_mm=2.0, ax_step_mm=1.0)
    fp_ref, fp_smp = calc_first_pass(ref, axial, lateral, **kw), calc_first_pass(smp, axial, lateral, **kw)
    inter = intersect_first_pass(fp_smp, fp_ref)
    np.testing.assert_array_equal(inter.usable_both_2d, fp_smp.usable_2d & fp_ref.usable_2d)
    ok = inter.usable_both_2d
    assert np.all(inter.f_low_both_2d[ok] >= np.fmax(fp_smp.f_low_2d, fp_ref.f_low_2d)[ok] - 1e-12)
    deep = fp_smp.depths > axial[1100, 0] + 2   # sample is noise-only there
    assert not inter.usable_both_2d[deep].any() and inter.usable_both_2d[fp_smp.depths < 20].all()
    # band of a given ROI (block centres inside a box at 5-15 mm, all lines)
    roi = roi_band_check(inter, [lateral.min(), lateral.max(), lateral.max(), lateral.min()], [5, 5, 15, 15])
    assert roi.all_rows_usable and roi.depths.min() >= 5 and roi.depths.max() <= 15
    assert roi.f_low_roi <= 4.5 and roi.f_high_roi >= 7.5   # sample 4-8 MHz within reference 3-9 MHz
    # different imaging depths: the shared leading rows are kept
    short = calc_first_pass(smp[:, :1000], axial[:1000], lateral[:1000], **kw)
    inter2 = intersect_first_pass(short, fp_ref)
    assert inter2.sample.depths.size == short.depths.size == inter2.reference.depths.size


def test_trim_keeps_block_arrays_when_counts_coincide():
    """n_blocks == n_depths must not trim the per-block arrays (review finding)."""
    from quantus.seg_loading.correlation_length.first_pass import _trim_depths
    rng = np.random.default_rng(3)
    fs, n_samples, n_lines = 40.0, 1500, 64   # 11 depths at 2.55 mm and 11 ten-line blocks
    rf = _band_limited_rf(rng, 1, n_samples, n_lines, fs, (4, 8))
    axial, lateral = _linear_positions(n_samples, n_lines, fs)
    fp = calc_first_pass(rf, axial, lateral, window_length_mm=2.0, block_width_mm=2.0, ax_step_mm=2.55)
    assert fp.depths.size == fp.block_start.size  # the coincidence this test needs
    t = _trim_depths(fp, fp.depths.size - 2)
    assert t.depths.size == fp.depths.size - 2 and t.block_x.shape == (fp.depths.size - 2, fp.block_start.size)
    np.testing.assert_array_equal(t.block_start, fp.block_start)
    np.testing.assert_array_equal(t.block_centre, fp.block_centre)


def test_sector_sample_at_another_depth_uses_the_reference_blocks():
    rng = np.random.default_rng(4)
    n_samples, n_lines, fs = 1200, 80, 40.0
    r = 20 + np.arange(n_samples) * (1.540 / 2 / fs)
    th = np.deg2rad(60 * (np.arange(n_lines) / (n_lines - 1) - 0.5))
    axial, lateral = r[:, None] * np.cos(th)[None] - 20, r[:, None] * np.sin(th)[None]
    ref = calc_first_pass(_band_limited_rf(rng, 2, n_samples, n_lines, fs, (4, 8)), axial, lateral, 2.0, 4.0,
                          ax_step_mm=1.0)
    short = 700  # sample acquired shallower: its mid-depth line spacing differs
    smp_rf = _band_limited_rf(rng, 1, short, n_lines, fs, (4, 8))
    own = calc_first_pass(smp_rf, axial[:short], lateral[:short], 2.0, 4.0, ax_step_mm=1.0)
    assert own.block_lines != ref.block_lines
    with pytest.raises(ValueError, match="grids differ"):
        intersect_first_pass(own, ref)
    smp = calc_first_pass(smp_rf, axial[:short], lateral[:short], 2.0, 4.0, ax_step_mm=1.0,
                          block_lines=ref.block_lines)
    inter = intersect_first_pass(smp, ref)
    assert inter.sample.depths.size == smp.depths.size == inter.reference.depths.size


def test_impossible_settings_raise_clear_errors():
    rng = np.random.default_rng(5)
    rf = rng.normal(size=(1, 300, 20))
    axial, lateral = _linear_positions(300, 20, 40.0)
    with pytest.raises(ValueError, match="A-lines"):
        calc_first_pass(rf, axial, lateral, 1.0, 50.0, ax_step_mm=1.0)       # block wider than the image
    with pytest.raises(ValueError, match="window"):
        calc_first_pass(rf, axial, lateral, 1.0, 1.0, ax_start_mm=5.0, ax_end_mm=2.0, ax_step_mm=1.0)


def test_with_usable_level_changes_only_usability():
    from quantus.seg_loading.correlation_length.first_pass import with_usable_level, MIN_SIGNAL_POWER_DB
    rng = np.random.default_rng(4)
    fs, n_samples, n_lines = 40.0, 1200, 64
    rf = _band_limited_rf(rng, 3, n_samples, n_lines, fs, (4, 8))
    rf[:, :, 40:] = 0.3 * rng.normal(size=(3, n_samples, n_lines - 40))   # weaker region
    axial, lateral = _linear_positions(n_samples, n_lines, fs)
    fp = calc_first_pass(rf, axial, lateral, window_length_mm=2.0, block_width_mm=2.0, ax_step_mm=1.0)
    same = with_usable_level(fp, MIN_SIGNAL_POWER_DB)
    assert np.array_equal(same.usable_2d, fp.usable_2d) and np.array_equal(same.usable, fp.usable)
    strict, loose = with_usable_level(fp, 15.0), with_usable_level(fp, 0.0)
    assert strict.usable_2d.sum() <= fp.usable_2d.sum() <= loose.usable_2d.sum()
    assert not np.any(strict.usable_2d & ~fp.usable_2d)                      # stricter only removes
    assert np.array_equal(strict.f_low_2d, fp.f_low_2d, equal_nan=True)      # band edges unchanged
    assert strict.min_signal_power_db == 15.0

"""Correlation-length results in samples / A-lines and the whole-number recommended windows."""

import numpy as np
import pytest

from quantus.seg_loading.correlation_length import CorrelationLengths, correlation_lengths


def _result(**kw):
    args = dict(
        depths_mm=np.array([10.0, 20.0, 30.0]),
        ax_corr_mm=np.array([[0.20, 0.22], [0.30, 0.30], [np.nan, np.nan]]),  # 2 frames
        lat_corr_mm=np.array([[0.26, 0.26], [0.39, 0.39], [0.52, np.nan]]),
        ax_spacing_mm=0.05,
        lat_spacing_mm=np.array([0.26, 0.26, 0.26]),
        window_samples=np.array([201, 201, 201]),
        window_lines=np.array([31, 31, 31]),
        threshold=0.5, window_length_mm=10.0, window_width_mm=8.0,
    )
    args.update(kw)
    return CorrelationLengths(**args)


def test_lengths_in_samples_and_lines():
    r = _result()
    np.testing.assert_allclose(r.ax_mean_samples[:2], [4.2, 6.0])
    np.testing.assert_allclose(r.lat_mean_lines, [1.0, 1.5, 2.0])
    np.testing.assert_allclose(r.lat_corr_lines[:, 0], [1.0, 1.5, 2.0])


def test_recommended_windows_round_up_to_whole_counts():
    r = _result()
    # axial 10 x 4.2 = 42 samples, 10 x 6.0 = 60 (no float-noise bump to 61), NaN stays NaN
    np.testing.assert_array_equal(r.recommended_ax_window_samples[:2], [42, 60])
    assert np.isnan(r.recommended_ax_window_samples[2])
    # lateral 5 x [1.0, 1.5, 2.0] = [5, 7.5 -> 8, 10]
    np.testing.assert_array_equal(r.recommended_lat_window_lines, [5, 8, 10])
    # mm: factor x mean length in mm, to the nearest 0.1 mm (5 x [0.26, 0.39, 0.52], 10 x [0.21, 0.30])
    np.testing.assert_array_equal(r.recommended_lat_window_mm, [1.3, 2.0, 2.6])  # 1.95 rounds half up
    np.testing.assert_array_equal(r.recommended_ax_window_mm[:2], [2.1, 3.0])
    assert np.isnan(r.recommended_ax_window_mm[2])
    r.window_mm_step = 0.5
    np.testing.assert_array_equal(r.recommended_lat_window_mm, [1.5, 2.0, 2.5])


def test_window_factors_are_inputs_and_recompute():
    r = _result(lat_window_factor=3.0, ax_window_factor=8.0)
    np.testing.assert_array_equal(r.recommended_lat_window_lines, [3, 5, 6])
    r.lat_window_factor = 4.0
    np.testing.assert_array_equal(r.recommended_lat_window_lines, [4, 6, 8])
    table = r.to_table()
    assert "lat_window_lines (4x)" in table.columns and len(table) == 3


def test_factors_pass_through_correlation_lengths():
    rng = np.random.default_rng(0)
    n_samples, n_lines = 600, 64
    rf = rng.normal(size=(2, n_samples, n_lines))
    axial = np.repeat(np.arange(n_samples)[:, None] * 0.05, n_lines, axis=1)
    lateral = np.repeat((np.arange(n_lines) - n_lines / 2)[None] * 0.3, n_samples, axis=0)
    r = correlation_lengths(rf, axial, lateral, 5.0, 6.0, 8.0, 20.0, 3,
                            ax_window_factor=7.0, lat_window_factor=3.0)
    assert (r.ax_window_factor, r.lat_window_factor) == (7.0, 3.0)
    ok = np.isfinite(r.lat_mean_lines)
    np.testing.assert_array_equal(r.recommended_lat_window_lines[ok], np.ceil(3.0 * r.lat_mean_lines[ok] - 1e-9))


def test_whole_range_window_is_the_largest_per_depth_recommendation():
    r = _result(lat_spacing_mm=np.array([0.2, 0.3, 0.4]))
    w = r.recommended_window()
    # axial: [42, 60, NaN] samples -> 60 at 20 mm; lateral lines [6.5->7, 6.5->7, 6.5->7]
    assert (w["ax_window_samples"], w["ax_at_depth_mm"]) == (60, 20.0)
    np.testing.assert_allclose(w["ax_window_mm"], 60 * 0.05)
    assert w["lat_window_lines"] == r.recommended_lat_window_lines[np.nanargmax(r.recommended_lat_window_lines)]
    assert w["depth_range_mm"] == (10.0, 30.0)
    assert "samples" in r.describe_window() and "A-lines" in r.describe_window()


def _positions(n_samples, n_lines, d_ax=0.05, d_lat=0.3, ax0=0.0):
    axial = np.repeat(ax0 + np.arange(n_samples)[:, None] * d_ax, n_lines, axis=1)
    lateral = np.repeat((np.arange(n_lines) - n_lines / 2)[None] * d_lat, n_samples, axis=0)
    return axial, lateral


def test_window_centres_snap_clamp_and_report_axial_positions():
    rng = np.random.default_rng(1)
    n_samples, n_lines = 400, 40
    rf = rng.normal(size=(1, n_samples, n_lines))
    # first sample at 3 mm (e.g. an RF start delay); 2 mm window = 41 samples, half = 20
    axial, lateral = _positions(n_samples, n_lines, ax0=3.0)
    r = correlation_lengths(rf, axial, lateral, 2.0, 3.0, 0.0, 100.0, 5)  # range beyond the image
    assert np.isclose(r.depths_mm[0], 3.0 + 20 * 0.05)                    # clamped to first full window
    assert r.depths_mm[-1] <= 3.0 + (n_samples - 1 - 20) * 0.05 + 1e-9     # last full window
    r = correlation_lengths(rf, axial, lateral, 2.0, 3.0, 7.03, 12.0, 4)
    assert np.isclose(r.depths_mm[0], 7.05)                                # nearest sample to 7.03 mm
    r = correlation_lengths(rf, axial, lateral, 2.0, 3.0)                  # None: whole image, every half window
    np.testing.assert_allclose(np.diff(r.depths_mm), 20 * 0.05)
    r = correlation_lengths(rf, axial, lateral, 2.0, 3.0, 8.0, 9.0, 10_000)  # more depths than samples: step 1
    np.testing.assert_allclose(np.diff(r.depths_mm), 0.05)


def test_crossing_is_exact_on_the_linear_segment():
    from quantus.seg_loading.correlation_length.correlation import _first_crossing_shift
    assert _first_crossing_shift(np.array([1.0, 0.6, 0.2, 0.0]), 0.5) == pytest.approx(1.25)
    assert _first_crossing_shift(np.array([1.0, 0.3, 0.1]), 0.5) == pytest.approx(1 / 1.4)
    assert _first_crossing_shift(np.array([1.0, 0.9, 0.8]), 0.5) is None


def test_spacings_are_straight_line_distances_for_tilted_sectors():
    from quantus.seg_loading.correlation_length.correlation import _sample_spacings_mm
    n_s, n_l, d_th, tilt = 50, 21, np.deg2rad(0.5), np.deg2rad(20)
    r = 40 + np.arange(n_s) * 0.05
    th = tilt + (np.arange(n_l) - 10) * d_th
    axial, lateral = r[:, None] * np.cos(th), r[:, None] * np.sin(th)
    d_ax, d_lat = _sample_spacings_mm(axial, lateral, 20, 10)
    assert d_ax == pytest.approx(0.05)
    assert d_lat == pytest.approx(2 * r[20] * np.sin(d_th / 2))      # chord between neighbouring lines
    assert lateral[20, 10] - lateral[20, 9] < 0.95 * d_lat           # the old x-difference was ~cos(20 deg) short


def test_map_centre_column_equals_the_depth_profile():
    from quantus.seg_loading.correlation_length import correlation_length_map
    rng = np.random.default_rng(2)
    n_samples, n_lines = 500, 60
    rf = rng.normal(size=(2, n_samples, n_lines))
    axial, lateral = _positions(n_samples, n_lines)
    prof = correlation_lengths(rf, axial, lateral, 3.0, 4.0, 5.0, 20.0, 4)
    m = correlation_length_map(rf, axial, lateral, 3.0, 4.0, 5.0, 20.0, 4, lat_step_lines=5)
    centre = int(np.nonzero(m.cols == _mid(n_lines))[0][0])
    np.testing.assert_allclose(m.depth_mm[:, centre], prof.depths_mm)
    np.testing.assert_allclose(m.ax_corr_mm[:, centre], prof.ax_corr_mm)
    np.testing.assert_allclose(m.lat_corr_mm[:, centre], prof.lat_corr_mm)
    assert np.all(np.diff(m.cols) == 5)
    # columns whose full window doesn't fit are NaN; the edges of a 4 mm (13-line) window
    half = (m.window_lines[0] - 1) // 2
    fits = (m.cols - half >= 0) & (m.cols + half < n_lines)
    assert np.all(np.isnan(m.lat_mean_lines[:, ~fits])) and np.all(np.isfinite(m.lat_mean_lines[:, fits]))
    w = m.recommended_window()
    assert w["lat_window_lines"] == np.nanmax(m.recommended_lat_window_lines)
    assert len(m.to_table()) == fits.sum() * len(m.rows)


def _mid(n_lines):
    return int(np.sign(n_lines / 2) * np.floor(abs(n_lines / 2) + 0.5)) - 1


def test_presets_fill_kwargs_and_explicit_kwargs_win():
    from quantus.seg_loading.correlation_length import PROBE_PRESETS
    from quantus.seg_loading.correlation_length.correlation import _apply_preset
    assert {p: v["depth_range_mm"] for p, v in PROBE_PRESETS.items()} == {
        "C3": (5.0, 95.0), "L15": (5.0, 35.0), "L20": (5.0, 16.0)}
    assert all(v["lateral_fraction"] == 0.5 and v["threshold"] == 0.2 for v in PROBE_PRESETS.values())
    k = _apply_preset("C3", {"threshold": 0.5})
    assert k["threshold"] == 0.5 and k["depth_range_mm"] == (5.0, 95.0) and k["lateral_fraction"] == 0.5
    assert _apply_preset(None, {"a": 1}) == {"a": 1}
    with pytest.raises(ValueError, match="Unknown preset"):
        _apply_preset("C5", {})


def test_depth_range_keeps_whole_windows_inside():
    rng = np.random.default_rng(9)
    n_samples, n_lines = 600, 40
    rf = rng.normal(size=(1, n_samples, n_lines))
    axial, lateral = _positions(n_samples, n_lines, ax0=1.0)  # 0.05 mm samples from 1 mm
    r = correlation_lengths(rf, axial, lateral, 4.0, 3.0, depth_range_mm=(5.0, 25.0))
    half = (r.window_samples[0] - 1) // 2 * 0.05  # 81 samples -> 2 mm either side of the centre
    assert r.depths_mm[0] - half >= 5.0 - 1e-9 and r.depths_mm[0] - half < 5.0 + 0.05
    assert r.depths_mm[-1] + half <= 25.0 + 1e-9
    # combined with centre limits, the stricter one wins
    r2 = correlation_lengths(rf, axial, lateral, 4.0, 3.0, ax_start_mm=10.0, depth_range_mm=(5.0, 25.0))
    assert np.isclose(r2.depths_mm[0], 10.0)
    with pytest.raises(ValueError, match="depth_range_mm"):
        correlation_lengths(rf, axial, lateral, 4.0, 3.0, depth_range_mm=(5.0, 7.0))


def test_lateral_positions_stay_inside_the_central_region():
    from quantus.seg_loading.correlation_length.correlation import _central_region, _lateral_positions
    first, last = _central_region(192, 95, 0.5)
    assert (first, last) == (47, 143)
    cols = _lateral_positions(95, 10, first, last)  # 21-line window, a centre every 10 lines
    assert 95 in cols and np.all(np.diff(cols) == 10)
    assert cols.min() - 10 >= first and cols.max() + 10 <= last
    assert list(_lateral_positions(95, 60, first, last)) == [95]  # window wider than the region: centre only
    assert _central_region(192, 95, None) == (0, 191)
    with pytest.raises(ValueError):
        _central_region(192, 95, 0.0)


def test_central_half_averages_frames_and_positions():
    rng = np.random.default_rng(8)
    n_samples, n_lines, n_frames = 500, 80, 2
    rf = rng.normal(size=(n_frames, n_samples, n_lines))
    axial, lateral = _positions(n_samples, n_lines)
    centre = correlation_lengths(rf, axial, lateral, 3.0, 3.0, 5.0, 20.0, 4)
    half = correlation_lengths(rf, axial, lateral, 3.0, 3.0, 5.0, 20.0, 4, lateral_fraction=0.5)
    np.testing.assert_allclose(half.depths_mm, centre.depths_mm)
    assert centre.n_positions is None and np.all(centre.n_measurements == n_frames)
    # 3 mm / 0.3 mm = 11 lines (half 5), central 40 lines -> positions mid0 + {-15,-10,...,15}? only those fitting
    from quantus.seg_loading.correlation_length.correlation import _central_region, _lateral_positions
    first, last = _central_region(n_lines, _mid(n_lines), 0.5)
    expected = _lateral_positions(_mid(n_lines), 5, first, last)
    assert np.all(half.n_positions == len(expected)) and len(expected) > 1
    assert half.n_frames == n_frames and half.ax_corr_mm.shape == (len(half.depths_mm), n_frames * len(expected))
    # the centre-line measurement is one of the columns (position index of the centre line, frame-major)
    p = int(np.nonzero(expected == _mid(n_lines))[0][0])
    np.testing.assert_allclose(half.lat_corr_mm[:, p::len(expected)], centre.lat_corr_mm)
    assert np.all(half.to_table()["n_measurements"] <= n_frames * len(expected))
    assert "lateral positions" in half.describe_window()
    # the map with the same fraction measures only windows inside the region
    from quantus.seg_loading.correlation_length import correlation_length_map
    m = correlation_length_map(rf, axial, lateral, 3.0, 3.0, 5.0, 20.0, 4, lateral_fraction=0.5)
    measured = np.isfinite(m.lat_mean_lines).any(axis=0)
    assert np.all(m.cols[measured] - 5 >= first) and np.all(m.cols[measured] + 5 <= last)


def test_map_region_limits_the_recommended_window():
    from quantus.seg_loading.correlation_length import correlation_length_map
    rng = np.random.default_rng(3)
    n_samples, n_lines = 500, 60
    rf = rng.normal(size=(1, n_samples, n_lines))
    axial, lateral = _positions(n_samples, n_lines)
    m = correlation_length_map(rf, axial, lateral, 3.0, 4.0, lat_step_lines=5)
    i, j = np.unravel_index(int(np.nanargmax(m.recommended_lat_window_lines)), m.depth_mm.shape)
    m.lat_corr_mm[i, j] *= 50  # an artefact (wall / reverberation) in one window
    whole = m.recommended_window()
    assert (whole["lat_at_depth_mm"], whole["lat_at_lateral_mm"]) == (m.depth_mm[i, j], m.lateral_mm[i, j])
    d = m.depth_mm[i, j]
    region = m.recommended_window(depth_range_mm=(d + 0.5, None))  # excludes that row
    assert region["lat_window_lines"] < whole["lat_window_lines"]
    assert region["depth_range_mm"][0] > d
    with pytest.raises(ValueError, match="No measured windows"):
        m.recommended_window(depth_range_mm=(1000, 2000))


def test_sd_is_zero_for_one_measurement_and_nan_for_none():
    """With lateral_fraction depths can have different numbers of measurements (review finding)."""
    r = _result(ax_corr_mm=np.array([[0.2, np.nan], [0.3, 0.32], [np.nan, np.nan]]),
                lat_corr_mm=np.array([[0.26, np.nan], [0.39, 0.4], [np.nan, np.nan]]))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sd, mean = r.ax_std_mm, r.ax_mean_mm
    assert sd[0] == 0.0 and sd[1] > 0 and np.isnan(sd[2]) and np.isnan(mean[2])


def test_line_range_measures_inside_an_roi():
    rng = np.random.default_rng(9)
    n_samples, n_lines = 500, 80
    rf = rng.normal(size=(2, n_samples, n_lines))
    axial, lateral = _positions(n_samples, n_lines)
    from quantus.seg_loading.correlation_length.correlation import _lateral_positions
    r = correlation_lengths(rf, axial, lateral, 3.0, 3.0, line_range=(10, 45), depth_range_mm=(6.0, 20.0))
    expected = _lateral_positions((10 + 45) // 2, 5, 10, 45)   # 11-line windows inside lines 10-45
    assert np.all(r.n_positions == len(expected)) and expected.min() - 5 >= 10 and expected.max() + 5 <= 45
    assert r.line_range == (10, 45) and r.lateral_fraction is None
    assert "A-lines 10-45" in r.describe_window()
    w = r.recommended_window()
    assert w["depth_range_mm"][0] >= 6.0 and w["depth_range_mm"][1] <= 20.0
    # same as lateral_fraction when the range is the central region
    from quantus.seg_loading.correlation_length.correlation import _central_region
    first, last = _central_region(n_lines, _mid(n_lines), 0.5)
    a = correlation_lengths(rf, axial, lateral, 3.0, 3.0, 5.0, 20.0, 4, lateral_fraction=0.5)
    b = correlation_lengths(rf, axial, lateral, 3.0, 3.0, 5.0, 20.0, 4, line_range=(first, last))
    assert (first + last) // 2 == _mid(n_lines)
    np.testing.assert_allclose(a.lat_corr_mm, b.lat_corr_mm)
    with pytest.raises(ValueError):
        correlation_lengths(rf, axial, lateral, 3.0, 3.0, line_range=(70, 90))

"""Correlation-length results in samples / A-lines and the whole-number recommended windows."""

import numpy as np
import pytest

from quantus.correlation_length import CorrelationLengths, correlation_lengths


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
    # mm are the whole counts x spacing
    np.testing.assert_allclose(r.recommended_lat_window_mm, [5 * 0.26, 8 * 0.26, 10 * 0.26])
    np.testing.assert_allclose(r.recommended_ax_window_mm[:2], [42 * 0.05, 60 * 0.05])


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
    from quantus.correlation_length.correlation import _first_crossing_shift
    assert _first_crossing_shift(np.array([1.0, 0.6, 0.2, 0.0]), 0.5) == pytest.approx(1.25)
    assert _first_crossing_shift(np.array([1.0, 0.3, 0.1]), 0.5) == pytest.approx(1 / 1.4)
    assert _first_crossing_shift(np.array([1.0, 0.9, 0.8]), 0.5) is None


def test_spacings_are_straight_line_distances_for_tilted_sectors():
    from quantus.correlation_length.correlation import _sample_spacings_mm
    n_s, n_l, d_th, tilt = 50, 21, np.deg2rad(0.5), np.deg2rad(20)
    r = 40 + np.arange(n_s) * 0.05
    th = tilt + (np.arange(n_l) - 10) * d_th
    axial, lateral = r[:, None] * np.cos(th), r[:, None] * np.sin(th)
    d_ax, d_lat = _sample_spacings_mm(axial, lateral, 20, 10)
    assert d_ax == pytest.approx(0.05)
    assert d_lat == pytest.approx(2 * r[20] * np.sin(d_th / 2))      # chord between neighbouring lines
    assert lateral[20, 10] - lateral[20, 9] < 0.95 * d_lat           # the old x-difference was ~cos(20 deg) short


def test_map_centre_column_equals_the_depth_profile():
    from quantus.correlation_length import correlation_length_map
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


def test_map_region_limits_the_recommended_window():
    from quantus.correlation_length import correlation_length_map
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

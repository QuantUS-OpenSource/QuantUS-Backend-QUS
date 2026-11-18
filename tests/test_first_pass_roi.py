"""Interactive first-pass ROI helpers (Python side of DrawROIWithBand.m)."""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from quantus.seg_loading.correlation_length.roi import (
    rf_outline_points, roi_spline_for_pkl, save_roi_pkl, sector_roi_from_box,
)


def _linear(n_samples=400, n_lines=60, d_ax=0.05, pitch=0.2):
    axial = np.repeat(np.arange(n_samples)[:, None] * d_ax, n_lines, axis=1)
    lateral = np.repeat(((np.arange(n_lines) - (n_lines - 1) / 2) * pitch)[None], n_samples, axis=0)
    return axial, lateral


def _sector(n_samples=500, n_lines=81, r0=20.0, d_r=0.1, width_deg=60.0):
    r = r0 + np.arange(n_samples) * d_r
    th = np.deg2rad(width_deg * (np.arange(n_lines) / (n_lines - 1) - 0.5))
    return r[:, None] * np.cos(th)[None] - r0, r[:, None] * np.sin(th)[None]


def test_box_on_a_linear_probe_is_the_rf_rectangle():
    axial, lateral = _linear()
    roi = sector_roi_from_box(axial, lateral, -2.0, 2.0, 5.0, 10.0)
    assert roi.rows == (100, 200)
    assert roi.lines == tuple(np.nonzero(np.abs(lateral[0]) <= 2.0)[0][[0, -1]])
    assert roi.depth_range_mm == pytest.approx((5.0, 10.0))
    m = roi.mask(*axial.shape)
    assert m.sum() == 101 * (roi.lines[1] - roi.lines[0] + 1)   # rows 100..200
    # box drawn upside down / right to left gives the same ROI
    assert sector_roi_from_box(axial, lateral, 2.0, -2.0, 10.0, 5.0).rows == roi.rows


def test_box_on_a_sector_follows_the_beam_lines():
    axial, lateral = _sector()
    roi = sector_roi_from_box(axial, lateral, -5.0, 5.0, 10.0, 30.0)
    (r1, r2), (l1, l2) = roi.rows, roi.lines
    rm = int(np.floor((r1 + r2) / 2 + 0.5))
    assert np.all(np.abs(lateral[rm, l1:l2 + 1]) <= 5.0)               # lines inside the box at mid depth
    assert abs(lateral[rm, l1 - 1]) > 5.0 and abs(lateral[rm, l2 + 1]) > 5.0
    # outline: top row, last line, bottom row, first line -> an annular sector, wider at the bottom
    assert np.ptp(roi.x_mm[roi.z_mm > 25]) > np.ptp(roi.x_mm[roi.z_mm < 12])
    # a box between lines snaps to the nearest one
    narrow = sector_roi_from_box(axial, lateral, 0.01, 0.02, 10.0, 30.0)
    assert narrow.lines[0] == narrow.lines[1]


def test_outline_points_and_pkl_round_trip_without_scan_conversion(tmp_path):
    axial, lateral = _linear()
    roi = sector_roi_from_box(axial, lateral, -2.0, 2.0, 5.0, 10.0)
    x, y = rf_outline_points(roi, max_per_side=20)
    assert x.min() == roi.lines[0] and x.max() == roi.lines[1] and y.min() == roi.rows[0] and y.max() == roi.rows[1]
    assert x.size <= 4 * 20
    image = SimpleNamespace(sc_bmode=None, scan_name="scan", phantom_name="phantom")
    sx, sy = roi_spline_for_pkl(image, roi)
    np.testing.assert_array_equal(sx, rf_outline_points(roi)[0])
    path = save_roi_pkl(tmp_path / "roi.pkl", image, roi, frame=0, extra={"Note": 1})
    info = pickle.load(open(path, "rb"))
    assert {"Spline X", "Spline Y", "Scan Name", "Phantom Name", "Frame", "RF Rows", "RF Lines", "Note"} <= set(info)
    assert info["RF Rows"] == roi.rows and info["Frame"] == 0
    # what pkl_roi does without scan conversion: polygon of the spline in (line, sample) space
    from PIL import Image, ImageDraw
    mask = Image.new("L", (axial.shape[1], axial.shape[0]), 0)
    ImageDraw.Draw(mask).polygon(list(zip(info["Spline X"], info["Spline Y"])), outline=1, fill=1)
    np.testing.assert_array_equal(np.array(mask).astype(bool), roi.mask(*axial.shape))


def test_scan_converted_spline_maps_back_to_the_roi_lines_and_samples():
    # toy scan conversion: sc pixel (y, x) -> RF line = x // 2, sample = 3 * y
    n_sc_y, n_sc_x = 200, 120
    yy, xx = np.mgrid[0:n_sc_y, 0:n_sc_x]
    image = SimpleNamespace(sc_bmode=np.ones((n_sc_y, n_sc_x)), xmap=xx // 2, ymap=3 * yy,
                            axial_res=0.05, sc_axial_res=0.15, scan_name="s", phantom_name="p")
    axial, lateral = _linear(n_samples=600, n_lines=60)
    roi = sector_roi_from_box(axial, lateral, -2.0, 2.0, 5.0, 20.0)
    sx, sy = roi_spline_for_pkl(image, roi)
    lines = image.xmap[sy.astype(int), sx.astype(int)]
    samples = image.ymap[sy.astype(int), sx.astype(int)]
    assert lines.min() == roi.lines[0] and lines.max() == roi.lines[1]
    assert abs(samples.min() - roi.rows[0]) <= 2 and abs(samples.max() - roi.rows[1]) <= 2  # 3 samples per sc pixel

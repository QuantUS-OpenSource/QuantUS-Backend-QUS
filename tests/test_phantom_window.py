"""Tests for the @phantom_window decorator: each window/full_segmentation analysis function
declares which slice of the phantom RF data it needs, and the framework does the slicing.
Uses small synthetic RF data (no sample scans ship with this repo).
"""

import numpy as np
import pytest

from quantus.analysis.options import get_analysis_types
from quantus.analysis.paramap.decorators import PHANTOM_WINDOW_POLICIES, phantom_window
from quantus.analysis.paramap.framework import ParamapAnalysis, get_phantom_window_policy
from quantus.data_objs.analysis import Window
from quantus.data_objs.analysis_config import RfAnalysisConfig, WIN_SIZE_UNITS_MM
from quantus.data_objs.image import UltrasoundRfImage
from quantus.data_objs.seg import BmodeSeg

HSCAN_KWARGS = dict(gh1_order=2, gh1_sigma=0.11e-6, gh1_wavelet_duration=3e-6,
                    gh2_order=8, gh2_sigma=0.09e-6, gh2_wavelet_duration=3e-6)


def _make_inputs(n_ax=600, n_lat=64, n_lat_phantom=96):
    rng = np.random.default_rng(0)
    image = UltrasoundRfImage("scan.raw", "phantom.raw")
    image.rf_data = rng.normal(size=(n_ax, n_lat))
    image.phantom_rf_data = rng.normal(size=(n_ax, n_lat_phantom))
    image.bmode = np.abs(image.rf_data)
    image.axial_res = 0.025
    image.lateral_res = 0.3
    image.spatial_dims = 2

    seg = BmodeSeg()
    seg.seg_mask = np.zeros((n_ax, n_lat), dtype=bool)
    seg.seg_mask[100:500, 10:54] = True
    seg.splines = []
    seg.frame = 0

    config = RfAnalysisConfig()
    config.transducer_freq_band = [1e6, 9e6]
    config.analysis_freq_band = [3e6, 7e6]
    config.center_frequency = 5e6
    config.sampling_frequency = 30e6
    config.ax_win_size = 2.5
    config.lat_win_size = 3.0
    config.ax_win_size_units = WIN_SIZE_UNITS_MM
    config.lat_win_size_units = WIN_SIZE_UNITS_MM
    config.speed_of_sound = 1540.
    config.window_thresh = 0.95
    config.axial_overlap = 0.5
    config.lateral_overlap = 0.5
    config.cor_win_size = None
    config.coronal_overlap = None
    return image, seg, config


def _window(ax_min, ax_max, lat_min, lat_max):
    window = Window()
    window.ax_min, window.ax_max, window.lat_min, window.lat_max = ax_min, ax_max, lat_min, lat_max
    return window


def test_every_discovered_function_declares_a_valid_policy():
    _, functions = get_analysis_types()
    for name, func in functions["paramap"].items():
        assert get_phantom_window_policy(func) in PHANTOM_WINDOW_POLICIES, name


def test_mixed_policies_hscan_and_attenuation_coef_run_together():
    """Regression: hscan divides scan by phantom element-wise, so it broke when every
    function received the central-lines phantom slice."""
    image, seg, config = _make_inputs()
    analysis = ParamapAnalysis(image, config, seg, ["hscan", "attenuation_coef"],
                               ref_atten=0.7, ref_atten_exp=1., ref_atten_offset=0.,
                               n_secs=5, sec_overlap=0.5, n_fft=256, **HSCAN_KWARGS)
    analysis.compute_paramaps()
    analysis.compute_single_window()
    assert len(analysis.windows) > 0
    assert np.isfinite(analysis.windows[0].results.ac_slope)
    # hscan is full_segmentation-only, so its output lives on the single ROI-spanning window
    single = analysis.single_window
    assert single.results.hscan_blue_channel.shape == (single.ax_max - single.ax_min + 1,
                                                       single.lat_max - single.lat_min + 1)


def test_matched_policy_matches_scan_window():
    image, seg, config = _make_inputs()
    analysis = ParamapAnalysis(image, config, seg, [])
    window = _window(100, 199, 10, 19)
    phantom = analysis.get_phantom_window(window, "matched")
    np.testing.assert_array_equal(phantom, image.phantom_rf_data[100:200, 10:20])


def test_central_lines_policy_ignores_window_lateral_position():
    image, seg, config = _make_inputs(n_lat_phantom=96)
    analysis = ParamapAnalysis(image, config, seg, [])
    left = analysis.get_phantom_window(_window(100, 199, 0, 9), "central_lines")
    right = analysis.get_phantom_window(_window(100, 199, 50, 59), "central_lines")
    np.testing.assert_array_equal(left, right)
    # MATLAB-style inclusive bounds: round(0.25*96) to round(0.75*96), 1-indexed -> lines 23..71
    np.testing.assert_array_equal(left, image.phantom_rf_data[100:200, 23:72])


def test_matched_policy_shifts_inward_when_phantom_is_narrower():
    image, seg, config = _make_inputs(n_lat=64, n_lat_phantom=40)
    analysis = ParamapAnalysis(image, config, seg, [])
    phantom = analysis.get_phantom_window(_window(100, 199, 50, 59), "matched")
    np.testing.assert_array_equal(phantom, image.phantom_rf_data[100:200, 30:40])


def test_matched_policy_raises_when_phantom_too_narrow_for_window():
    image, seg, config = _make_inputs(n_lat=64, n_lat_phantom=8)
    analysis = ParamapAnalysis(image, config, seg, [])
    with pytest.raises(ValueError, match="Phantom has 8 lines"):
        analysis.get_phantom_window(_window(100, 199, 0, 9), "matched")


def test_invalid_policy_rejected():
    @phantom_window("not_a_policy")
    def bad(*args, **kwargs):
        pass

    @phantom_window("matched", "central_lines")
    def two_policies(*args, **kwargs):
        pass

    for func in (bad, two_policies):
        with pytest.raises(ValueError, match="@phantom_window must be exactly one of"):
            get_phantom_window_policy(func)


def test_undeclared_policy_defaults_to_matched():
    def plain(*args, **kwargs):
        pass

    assert get_phantom_window_policy(plain) == "matched"

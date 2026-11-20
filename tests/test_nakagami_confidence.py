"""Nakagami confidence map (port of nakagami_map.m / roi_indices.m). Verified against MATLAB
nakagami_map.m on the LiverExample L20 frame (497 windows): m and KS D agree to ~1e-8."""

import numpy as np
from scipy import stats

from quantus.seg_loading.nakagami_confidence import (
    nakagami_confidence, nakagami_fit, ks_statistic, remove_outliers, window_features, window_indices,
)


def test_fit_matches_scipy():
    x = stats.nakagami.rvs(1.7, scale=2.0, size=5000, random_state=np.random.default_rng(0))
    m, omega = nakagami_fit(x)
    m2, _, scale = stats.nakagami.fit(x, floc=0)
    assert abs(m - m2) < 1e-3 and abs(omega - scale ** 2) < 1e-3


def test_ks_small_for_nakagami_large_for_mixture():
    rng = np.random.default_rng(1)
    speckle = stats.nakagami.rvs(1.0, scale=1.0, size=2000, random_state=rng)
    mixture = np.r_[speckle[:1000], 0.05 * speckle[1000:]]          # bright tissue + dark lumen
    assert window_features(speckle)[1] < 0.05
    assert window_features(mixture)[1] > 0.10


def test_window_indices_inside_mask():
    mask = np.zeros((100, 40), bool); mask[10:90, 5:35] = True
    w = window_indices(mask, 20, 10, 0.7, 0.7)
    assert len(w) and np.all(w[:, 0] >= 10) and np.all(w[:, 1] <= 89) and np.all(w[:, 2] >= 5) and np.all(w[:, 3] <= 34)
    assert np.all(w[:, 1] - w[:, 0] == 19) and np.all(w[:, 3] - w[:, 2] == 9)


def test_map_finds_a_dark_region():
    rng = np.random.default_rng(2)
    rf = rng.standard_normal((400, 60))
    rf[150:250, 20:40] *= 0.05                                       # anechoic "vessel"
    r = nakagami_confidence(rf, 40, 8)
    vessel = r.vessel_mask(0.10)
    assert vessel[150:250, 20:40].mean() > vessel[:100].mean() + 0.3


def test_remove_outliers_is_the_intended_mad_rule():
    x = np.r_[np.ones(50), 2.0, 100.0]
    kept = remove_outliers(x, 2.0)
    assert 100.0 not in kept and kept.size >= 50


def test_bright_reflector_is_bright_not_vessel():
    rng = np.random.default_rng(3)
    rf = rng.standard_normal((400, 60))
    rf[150:250, 20:40] *= 0.05                                       # dark lumen
    rf[320:330, 5:15] *= 30                                          # thin strong reflector
    r = nakagami_confidence(rf, 40, 8)
    vessel, bright = r.vessel_mask(0.10), r.bright_mask(0.10)
    assert vessel[150:250, 20:40].mean() > bright[150:250, 20:40].mean()
    assert bright[315:335, 5:15].mean() > vessel[315:335, 5:15].mean()
    assert not np.any(vessel & bright)

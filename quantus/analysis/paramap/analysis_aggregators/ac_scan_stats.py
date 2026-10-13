from typing import List

import numpy as np
from scipy.optimize import curve_fit

from ..decorators import *
from ....data_objs.analysis import Window, BlankResults
from ....data_objs.analysis_config import RfAnalysisConfig
from ....data_objs.image import UltrasoundRfImage

def power1_model(x, a, b):
    return a * (x ** b)

@supported_spatial_dims(2, 3)
@output_vars("computed_sample_AC", "AC_power1_a", "AC_power1_b")
@dependencies("attenuation_coef")
@location("aggregate")
def ac_scan_stats(windows: List[Window], aggregate_results: BlankResults, config: RfAnalysisConfig,
                   image_data: UltrasoundRfImage, **kwargs) -> None:
    """Aggregate the per-window attenuation-coefficient-vs-frequency curves (from
    "attenuation_coef") into scan-wide AC estimates: average the curve across every window
    in the ROI first, then reduce it to scalars. Mirrors the MATLAB reference driver's final
    AC step (test_runner_main_loop3.m, Division_AC/Power1_AC columns), which needs every
    window's curve simultaneously and so can't run as a per-window analysis function.

    Writes:
        computed_sample_AC: mean(att_ave / att_subf) -- simple division-based AC estimate.
        AC_power1_a, AC_power1_b: power-law fit att_ave = a * f^b via scipy.optimize.curve_fit.
    """
    params = windows[0].results.__dict__.keys()
    assert "att_coefs" in params and "att_subf" in params, "Must run \"attenuation_coef\" plugin to use this"

    subf = windows[0].results.att_subf
    att_all = np.array([w.results.att_coefs for w in windows]).T
    att_ave = np.mean(att_all, axis=1)

    aggregate_results.computed_sample_AC = np.mean(att_ave / subf)

    popt, _ = curve_fit(power1_model, subf, att_ave)
    aggregate_results.AC_power1_a, aggregate_results.AC_power1_b = popt

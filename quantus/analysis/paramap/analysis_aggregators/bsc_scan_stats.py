from typing import List

import numpy as np

from ..decorators import *
from ....data_objs.analysis import Window, BlankResults
from ....data_objs.analysis_config import RfAnalysisConfig
from ....data_objs.image import UltrasoundRfImage


@supported_spatial_dims(2, 3)
@output_vars("bsc_slope", "bsc_intercept", "bsc_midband_fit")
@dependencies("bsc_transmission_compensation")
@location("aggregate")
def bsc_scan_stats(windows: List[Window], aggregate_results: BlankResults, config: RfAnalysisConfig,
                    image_data: UltrasoundRfImage, **kwargs) -> None:
    """Lizzi-Feleppa slope/intercept/midband-fit of the scan-averaged BSC-vs-frequency
    curve (from "bsc_transmission_compensation"). Mirrors the MATLAB reference driver's
    post-loop BSC curve-fit step (test_runner_main_loop3.m, ~lines 489-510):
    average the BSC curve across every window first, convert to dB, then fit a robust
    (outlier-trimmed) line to it -- needs every window's curve simultaneously, so this
    can't run as a per-window analysis function.
    """
    params = windows[0].results.__dict__.keys()
    assert "bsc_curve" in params and "bsc_freq_mhz" in params, \
        "Must run \"bsc_transmission_compensation\" plugin to use this"

    freq = windows[0].results.bsc_freq_mhz
    mean_bsc = np.nanmean(np.array([w.results.bsc_curve for w in windows]), axis=0)

    valid = (mean_bsc > 0) & np.isfinite(mean_bsc)
    freq_fit = freq[valid]
    bsc_db = 10 * np.log10(mean_bsc[valid])  # 10*log10 (power quantity), not 20*log10

    p = np.polyfit(freq_fit, bsc_db, 1)
    err = np.abs(np.polyval(p, freq_fit) - bsc_db)
    keep = err <= np.median(err)
    p = np.polyfit(freq_fit[keep], bsc_db[keep], 1)  # second-pass, outlier-robust fit

    aggregate_results.bsc_slope = p[0]      # dB/MHz
    aggregate_results.bsc_intercept = p[1]  # dB
    # Median of the FULL freq_fit (not the outlier-trimmed subset), matching MATLAB exactly.
    aggregate_results.bsc_midband_fit = np.polyval(p, np.median(freq_fit))

from typing import List

import numpy as np

from ..decorators import *
from ....data_objs.analysis import Window, BlankResults
from ....data_objs.analysis_config import RfAnalysisConfig
from ....data_objs.image import UltrasoundRfImage


@supported_spatial_dims(2)
@output_vars("hk_k_mean", "hk_mu_mean")
@dependencies("homodyned_k_params")
@location("aggregate")
def homodyned_k_scan_stats(windows: List[Window], aggregate_results: BlankResults, config: RfAnalysisConfig,
                            image_data: UltrasoundRfImage, **kwargs) -> None:
    """Aggregate the per-window homodyned K-distribution estimates (from
    "homodyned_k_params") into scan-wide means, mirroring the MATLAB reference driver's
    post-loop step (test_runner_main_loop3.m: k_mean = mean(k_vals,'omitnan'),
    mu_mean = mean(mu_vals,'omitnan')).
    """
    params = windows[0].results.__dict__.keys()
    assert "hk_k" in params and "hk_mu" in params, "Must run \"homodyned_k_params\" plugin to use this"

    aggregate_results.hk_k_mean = np.nanmean([w.results.hk_k for w in windows])
    aggregate_results.hk_mu_mean = np.nanmean([w.results.hk_mu for w in windows])

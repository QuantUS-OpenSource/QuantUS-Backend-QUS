from typing import Dict

from ..decorators import required_kwargs
from ....data_objs.visualizations import ParamapDrawingBase

@required_kwargs()
def bsc_lf_stats(visualizations_obj: ParamapDrawingBase, data_dict: Dict[str, str], **kwargs) -> None:
    """Export the Lizzi-Feleppa BSC slope/intercept/midband-fit values. The actual
    cross-window averaging and curve fit happens in the "bsc_scan_stats" aggregate
    analysis function (quantus/analysis/paramap/analysis_aggregators/bsc_scan_stats.py), since
    it needs every window's BSC curve simultaneously -- this function just reads the result.
    """
    aggregate_results = visualizations_obj.analysis_obj.aggregate_results
    assert hasattr(aggregate_results, "bsc_slope") and hasattr(aggregate_results, "bsc_intercept") \
        and hasattr(aggregate_results, "bsc_midband_fit"), \
        "Must run \"bsc_scan_stats\" aggregate analysis function to use this"

    data_dict["bsc_slope"] = aggregate_results.bsc_slope
    data_dict["bsc_intercept"] = aggregate_results.bsc_intercept
    data_dict["bsc_midband_fit"] = aggregate_results.bsc_midband_fit

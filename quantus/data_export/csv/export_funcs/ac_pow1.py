from typing import Dict

from ..decorators import required_kwargs
from ....data_objs.visualizations import ParamapDrawingBase

@required_kwargs()
def ac_pow1(visualizations_obj: ParamapDrawingBase, data_dict: Dict[str, str], **kwargs) -> None:
    """Export sample AC values computed using the power1 model for curve fitting. The actual
    cross-window averaging and curve fit happens in the "ac_scan_stats" aggregate analysis
    function (quantus/analysis/paramap/analysis_aggregators/ac_scan_stats.py), since it needs
    every window's AC-vs-frequency curve simultaneously -- this function just reads the result.
    """
    aggregate_results = visualizations_obj.analysis_obj.aggregate_results
    assert hasattr(aggregate_results, "AC_power1_a") and hasattr(aggregate_results, "AC_power1_b"), \
        "Must run \"ac_scan_stats\" aggregate analysis function to use this"

    data_dict["AC_power1_a"] = aggregate_results.AC_power1_a
    data_dict["AC_power1_b"] = aggregate_results.AC_power1_b

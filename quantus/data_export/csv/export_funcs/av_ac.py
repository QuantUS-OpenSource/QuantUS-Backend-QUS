from typing import Dict

from ..decorators import required_kwargs
from ....data_objs.visualizations import ParamapDrawingBase

@required_kwargs()
def av_ac(visualizations_obj: ParamapDrawingBase, data_dict: Dict[str, str], **kwargs) -> None:
    """Export the average AC value (division-based estimate). The actual cross-window
    averaging and computation happens in the "ac_scan_stats" aggregate analysis function
    (quantus/analysis/paramap/analysis_aggregators/ac_scan_stats.py), since it needs every
    window's AC-vs-frequency curve simultaneously -- this function just reads the result.
    """
    aggregate_results = visualizations_obj.analysis_obj.aggregate_results
    assert hasattr(aggregate_results, "computed_sample_AC"), \
        "Must run \"ac_scan_stats\" aggregate analysis function to use this"

    data_dict["computed_sample_AC"] = [aggregate_results.computed_sample_AC]

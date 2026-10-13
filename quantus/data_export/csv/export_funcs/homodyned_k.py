from typing import Dict

from ..decorators import required_kwargs
from ....data_objs.visualizations import ParamapDrawingBase

@required_kwargs()
def homodyned_k(visualizations_obj: ParamapDrawingBase, data_dict: Dict[str, str], **kwargs) -> None:
    """Export the scan-wide homodyned K-distribution parameter means. The actual
    cross-window averaging happens in the "homodyned_k_scan_stats" aggregate analysis
    function (quantus/analysis/paramap/analysis_aggregators/homodyned_k_scan_stats.py), since
    it needs every window's k/mu estimate simultaneously -- this function just reads the
    result.
    """
    aggregate_results = visualizations_obj.analysis_obj.aggregate_results
    assert hasattr(aggregate_results, "hk_k_mean") and hasattr(aggregate_results, "hk_mu_mean"), \
        "Must run \"homodyned_k_scan_stats\" aggregate analysis function to use this"

    data_dict["hk_k_mean"] = [aggregate_results.hk_k_mean]
    data_dict["hk_mu_mean"] = [aggregate_results.hk_mu_mean]

from typing import List

import numpy as np

from ..decorators import *
from ....data_objs.analysis import Window, BlankResults
from ....data_objs.analysis_config import RfAnalysisConfig
from ....data_objs.image import UltrasoundRfImage


def _gather(windows: List[Window], source_attr: str, source_func: str) -> np.ndarray:
    assert hasattr(windows[0].results, source_attr), \
        f"'{source_attr}' not found on window results -- is '{source_func}' in analysis_funcs?"
    return np.array([getattr(w.results, source_attr) for w in windows])


@supported_spatial_dims(2, 3)
@output_vars()
@required_kwargs("source_func", "source_attr")
@location("aggregate")
def window_mean(windows: List[Window], aggregate_results: BlankResults, config: RfAnalysisConfig,
                 image_data: UltrasoundRfImage, **kwargs) -> None:
    """Generic aggregate function: mean of any window-level output across all windows.

    Reusable against any per-window plugin's scalar or equal-shape-array output, configured
    entirely via kwargs rather than a bespoke per-plugin aggregate function, e.g.:
        analysis_kwargs:
          source_func: nakagami_params   # window plugin that produced the output
          source_attr: nak_w             # attribute on window.results to average

    Writes aggregate_results.<source_attr>_mean. The output name is dynamic (depends on the
    source_attr kwarg) so it isn't statically declared via @output_vars -- a config pairing
    this with a data-export function needs to know the "<source_attr>_mean" convention.

    Note: analysis_kwargs is a single flat dict shared by every requested analysis function
    in a run, so only one source_func/source_attr pair can be aggregated per run today. Using
    window_mean/window_median against two different attributes in the same run isn't
    supported without a future kwargs-namespacing enhancement.
    """
    source_attr = kwargs["source_attr"]
    source_func = kwargs["source_func"]
    vals = _gather(windows, source_attr, source_func)
    setattr(aggregate_results, f"{source_attr}_mean", np.nanmean(vals, axis=0))


@supported_spatial_dims(2, 3)
@output_vars()
@required_kwargs("source_func", "source_attr")
@location("aggregate")
def window_median(windows: List[Window], aggregate_results: BlankResults, config: RfAnalysisConfig,
                   image_data: UltrasoundRfImage, **kwargs) -> None:
    """Generic aggregate function: median of any window-level output across all windows.
    See window_mean's docstring for the source_func/source_attr configuration convention and
    its single-pair-per-run limitation.

    Writes aggregate_results.<source_attr>_median.
    """
    source_attr = kwargs["source_attr"]
    source_func = kwargs["source_func"]
    vals = _gather(windows, source_attr, source_func)
    setattr(aggregate_results, f"{source_attr}_median", np.nanmedian(vals, axis=0))

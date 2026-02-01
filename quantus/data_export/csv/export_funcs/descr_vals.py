from typing import Dict

import numpy as np

from ..decorators import required_kwargs
from ....data_objs.visualizations import ParamapDrawingBase

@required_kwargs()
def descr_vals(visualizations_obj: ParamapDrawingBase, data_dict: Dict[str, str], **kwargs) -> None:
    """Compute descriptive values for each parameter in the analysis object and save to a CSV file.
    This includes mean, median, standard deviation, minimum, and maximum values.

    Args:
        visualizations_obj (ParamapDrawingBase): The visualizations object containing the data.
        params (Dict[str, str]): Dictionary of parameters to compute mean values for.
    """
    params = visualizations_obj.analysis_obj.windows[0].results.__dict__.keys()
    for param in params:
        if isinstance(getattr(visualizations_obj.analysis_obj.windows[0].results, param), (str, list, np.ndarray)):
            continue
        param_arr = [getattr(window.results, param) for window in visualizations_obj.analysis_obj.windows]
        data_dict[f"mean_{param}"] = [np.mean(param_arr)]
        data_dict[f"std_{param}"] = [np.std(param_arr)]
        data_dict[f"min_{param}"] = [np.min(param_arr)]
        data_dict[f"max_{param}"] = [np.max(param_arr)]
        data_dict[f"median_{param}"] = [np.median(param_arr)]

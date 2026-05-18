from typing import Dict

import numpy as np

from ..decorators import required_kwargs
from ....data_objs.visualizations import ParamapDrawingBase

@required_kwargs()
def av_ac(visualizations_obj: ParamapDrawingBase, data_dict: Dict[str, str], **kwargs) -> None:
    """Export the average AC values by taking the full AC curves from each window, averaging,
    and then performing curve fitting.s
    """
    params = visualizations_obj.analysis_obj.windows[0].results.__dict__.keys()
    
    assert "att_coefs" in params and "att_subf" in params, "Must run \"attenuation_coef\" plugin to use this"
    
    subf = visualizations_obj.analysis_obj.windows[0].results.att_subf
    att_all = np.array([w.results.att_coefs for w in visualizations_obj.analysis_obj.windows]).T
    att_ave = np.mean(att_all, axis=1)
    sample_atten = np.mean((att_ave) / subf)
    data_dict["computed_sample_AC"] = [sample_atten]

from typing import Dict

import numpy as np
from scipy.optimize import curve_fit

from ..decorators import required_kwargs
from ....data_objs.visualizations import ParamapDrawingBase

def power1_model(x, a, b):
    return a * (x ** b)

@required_kwargs()
def ac_pow1(visualizations_obj: ParamapDrawingBase, data_dict: Dict[str, str], **kwargs) -> None:
    """Export sample AC values computed using the power1 model for curve fitting.
    """
    params = visualizations_obj.analysis_obj.windows[0].results.__dict__.keys()
    
    assert "att_coefs" in params and "att_subf" in params, "Must run \"attenuation_coef\" plugin to use this"
    
    subf = visualizations_obj.analysis_obj.windows[0].results.att_subf
    att_all = np.array([w.results.att_coefs for w in visualizations_obj.analysis_obj.windows]).T
    att_ave = np.mean(att_all, axis=1)
    
    popt, pcov = curve_fit(power1_model, subf, att_ave)
    sample_atten_a, sample_atten_b = popt
    
    data_dict["AC_power1_a"] = sample_atten_a
    data_dict["AC_power1_b"] = sample_atten_b

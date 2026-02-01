import pickle
from pathlib import Path

from ..decorators import *
from ....data_objs.analysis_config import RfAnalysisConfig

@extensions(".pkl", ".pickle")
@gui_kwargs("assert_scan", "assert_phantom")
@default_gui_kwarg_vals("False", "False")       # str for GUI parsing
def pkl_rf(analysis_path: str, **kwargs) -> RfAnalysisConfig:
    """Function to load RF analysis configuration data from a pickle file saved from the QuantUS UI.
    
    Kwargs:
        assert_scan (bool): If True, assert that the scan file name matches the analysis config file name.
        assert_phantom (bool): If True, assert that the phantom file name matches the analysis config file name.
    """
    with open(analysis_path, "rb") as f:
        config_pkl: dict = pickle.load(f)
            
    if kwargs.get("assert_scan"):
        assert config_pkl["Image Name"] == Path(kwargs["scan_path"]).name, 'Scan file name mismatch'
    if kwargs.get("assert_phantom"):
        assert config_pkl["Phantom Name"] == Path(kwargs["phantom_path"]).name, 'Phantom file name mismatch'
    
    out = config_pkl["Config"]
    
    return out
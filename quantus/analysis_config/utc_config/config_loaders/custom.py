from ..decorators import *
from ....data_objs.analysis_config import RfAnalysisConfig

@extensions()
def custom(analysis_path: str, **kwargs) -> RfAnalysisConfig:
    """Function to load RF analysis configuration data from a Python dict.
    
    Kwargs:
        transducer_freq_band (list): [min, max] frequency band of the transducer (Hz).
        analysis_freq_band (list): [lower, upper] frequency band for analysis (Hz).
        center_frequency (float): Center frequency of the transducer (Hz).
        sampling_frequency (float): Sampling frequency (Hz).
        ax_win_size (float): Axial length per window (mm).
        lat_win_size (float): Lateral length per window (mm).
        window_thresh (float): Percentage of window area required to be considered in ROI.
        axial_overlap (float): Percentage of window overlap in axial direction.
        lateral_overlap (float): Percentage of window overlap in lateral direction.

        OPTIONAL FOR 3D SCANS:
        cor_win_size (float): Coronal length per window (mm).
        coronal_overlap (float): Percentage of window overlap in coronal direction.
    """
    out = RfAnalysisConfig()
    
    try:
        assert type(kwargs["analysis_freq_band"]) is list, "analysis_freq_band must be a list"
        assert len(kwargs["analysis_freq_band"]) == 2, "analysis_freq_band must be a list of two elements [lower, upper]"
        assert type(kwargs["transducer_freq_band"]) is list, "transducer_freq_band must be a list"
        assert len(kwargs["transducer_freq_band"]) == 2, "transducer_freq_band must be a list of two elements [min, max]"
        assert type(kwargs["center_frequency"]) is int, "center_frequency must be a int"
        assert type(kwargs["sampling_frequency"]) is int, "sampling_frequency must be a int"
        assert type(kwargs["ax_win_size"]) is float, "ax_win_size must be a float"
        assert type(kwargs["lat_win_size"]) is float, "lat_win_size must be a float"
        assert type(kwargs["window_thresh"]) is float, "window_thresh must be a float"
        assert type(kwargs["axial_overlap"]) is float, "axial_overlap must be a float"
        assert type(kwargs["lateral_overlap"]) is float, "lateral_overlap must be a float"

        out.transducer_freq_band = kwargs["transducer_freq_band"]
        out.analysis_freq_band = kwargs["analysis_freq_band"]
        out.center_frequency = kwargs["center_frequency"]
        out.sampling_frequency = kwargs["sampling_frequency"]
        out.ax_win_size = kwargs["ax_win_size"]
        out.lat_win_size = kwargs["lat_win_size"]
        out.window_thresh = kwargs["window_thresh"]
        out.axial_overlap = kwargs["axial_overlap"]
        out.lateral_overlap = kwargs["lateral_overlap"]
        out.cor_win_size = kwargs.get("cor_win_size", None)
        out.coronal_overlap = kwargs.get("coronal_overlap", None)

        if out.cor_win_size is not None:
            assert type(out.cor_win_size) is float, "cor_win_size must be a float"
            assert type(out.coronal_overlap) is float, "coronal_overlap must be a float"
            
    except KeyError as e:
        raise KeyError(f"Missing required key: {e}. Please provide all necessary parameters for the custom configuration.")

    return out
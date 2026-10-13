import numpy as np

from ..transforms import compute_power_spec
from ..decorators import *
from ....data_objs.analysis_config import RfAnalysisConfig
from ....data_objs.analysis import Window
from ....data_objs.image import UltrasoundRfImage

@supported_spatial_dims(2, 3)
@output_vars("f", "nps", "r_ps", "ps")
@required_kwargs("n_fft")
@default_kwarg_vals(8192)
def compute_power_spectra(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
                    window: Window, config: RfAnalysisConfig, 
                    image_data: UltrasoundRfImage, **kwargs) -> None:
    """Compute power spectra for a single window.
    """
    n_fft = kwargs['n_fft']
    f, ps = compute_power_spec(
        scan_rf_window, config.transducer_freq_band[0],
        config.transducer_freq_band[1], config.sampling_frequency,
        n_fft
    )
    ps = 20 * np.log10(ps)
    f, rPs = compute_power_spec(
        phantom_rf_window, config.transducer_freq_band[0],
        config.transducer_freq_band[1], config.sampling_frequency,
        n_fft
    )
    rPs = 20 * np.log10(rPs)
    nps = np.asarray(ps) - np.asarray(rPs)
    # Fill in attributes defined in ResultsClass above
    window.results.nps = nps # dB
    window.results.f = f # Hz
    window.results.ps = ps # dB
    window.results.r_ps = rPs # dB
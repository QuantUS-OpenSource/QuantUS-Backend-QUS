import numpy as np
from typing import Tuple

from ..decorators import *
from ....data_objs.analysis_config import RfAnalysisConfig
from ....data_objs.analysis import Window
from ....data_objs.image import UltrasoundRfImage

@supported_spatial_dims(2, 3)
@output_vars("mbf", "ss", "si")
@dependencies("compute_power_spectra")
def lizzi_feleppa(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
                    window: Window, config: RfAnalysisConfig, 
                    image_data: UltrasoundRfImage, **kwargs) -> None:
    """Compute spectral analysis values for a single window.
    
    Args:
        scan_rf_window (np.ndarray): RF data of the window in the scan image.
        phantom_rf_window (np.ndarray): RF data of the window in the phantom image.
        window (Window): Window object to store results.
        config (RfAnalysisConfig): Configuration object for analysis.
    """
    # Accessible from above function call
    nps = window.results.nps
    f = window.results.f
    
    def compute_spectral_params(nps: np.ndarray, f: np.ndarray, 
                               low_f: int, high_f: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Perform spectral analysis on the normalized power spectrum.
        source: Lizzi et al. https://doi.org/10.1016/j.ultrasmedbio.2006.09.002
        
        Args:
            nps (np.ndarray): normalized power spectrum.
            f (np.ndarray): frequency array (Hz).
            low_f (int): lower bound of the frequency window for analysis (Hz).
            high_f (int): upper bound of the frequency window for analysis (Hz).
            
        Returns:
            Tuple: midband fit, frequency range, linear fit, and linear regression coefficients.
        """
        # 1. in one scan / run-through of data file's f array, find the data points on
        # the frequency axis closest to reference file's analysis window's LOWER bound and UPPER bounds
        smallest_diff_low_f = 999999999
        smallest_diff_high_f = 999999999

        for i in range(len(f)):
            current_diff_low_f = abs(low_f - f[i])
            current_diff_high_f = abs(high_f - f[i])

            if current_diff_low_f < smallest_diff_low_f:
                smallest_diff_low_f = current_diff_low_f
                smallest_diff_index_low_f = i

            if current_diff_high_f < smallest_diff_high_f:
                smallest_diff_high_f = current_diff_high_f
                smallest_diff_index_high_f = i

        # 2. compute linear regression within the analysis window
        f_band = f[
            smallest_diff_index_low_f:smallest_diff_index_high_f
        ]  # transpose row vector f in order for it to have same dimensions as column vector nps
        p = np.polyfit(
            f_band, nps[smallest_diff_index_low_f:smallest_diff_index_high_f], 1
        )
        nps_linfit = np.polyval(p, f_band)  # y_linfit is a column vecotr

        mbfit = p[0] * f_band[round(f_band.shape[0] / 2)] + p[1]

        return mbfit, f_band, nps_linfit, p
    
    mbf, _, _, p = compute_spectral_params(nps, f, config.analysis_freq_band[0], config.analysis_freq_band[1])
    
    # Fill in attributes defined in "output_vars" decorator
    window.results.mbf = mbf # dB
    window.results.ss = p[0]*1e6 # dB/MHz
    window.results.si = p[1] # dB
from typing import List

# Named window size units, used for ax_win_size_units/lat_win_size_units below instead of
# unlabeled 1/2/3 codes.
WIN_SIZE_UNITS_MM = "mm"
WIN_SIZE_UNITS_WAVELENGTH = "number of wavelength"
WIN_SIZE_UNITS_A_LINES = "A-Lines"

class RfAnalysisConfig:
    """
    Class to store configuration data for RF analysis.
    """

    def __init__(self):
        self.transducer_freq_band: List[int]  # [min, max] (Hz)
        self.analysis_freq_band: List[int]  # [lower, upper] (Hz)
        self.sampling_frequency: int  # Hz

        self.ax_win_size_units: str # WIN_SIZE_UNITS_MM or WIN_SIZE_UNITS_WAVELENGTH
        self.lat_win_size_units: str # WIN_SIZE_UNITS_MM, WIN_SIZE_UNITS_WAVELENGTH, or WIN_SIZE_UNITS_A_LINES
        self.ax_win_size: float
        self.lat_win_size: float  # lateral width per window (mm)
        
        self.speed_of_sound: float # speed of sound through tissue
        self.window_thresh: float  # % of window area required to be in ROI
        self.axial_overlap: float  # % of ax window length to move before next window
        self.lateral_overlap: float  # % of lat window length to move before next window
        self.center_frequency: float  # Hz
        
        # 3D scan parameters
        self.cor_win_size: float  # coronal length per window (mm)
        self.coronal_overlap: float # % of cor window length to move before next window

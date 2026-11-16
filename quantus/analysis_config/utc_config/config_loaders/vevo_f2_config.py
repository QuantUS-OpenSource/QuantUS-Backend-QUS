from ..decorators import *
from ....data_objs.analysis_config import RfAnalysisConfig, WIN_SIZE_UNITS_WAVELENGTH
from ....image_loading.utc_loaders.vevo_f2_rf.parser import read_f2
from ....power_spectrum import compare_spectra

@extensions()
def vevo_f2_config(analysis_path: str, **kwargs) -> RfAnalysisConfig:
    """RF analysis configuration for Vevo F2 data (use with the vevo_f2_rf scan loader).

    The sampling frequency is derived from the scan's .rf.xml depth axis (as readF2.m does),
    and the analysis band / center frequency come from the overlap of the scan and phantom
    power spectra at bw_cutoff dB (as quantus.power_spectrum.compare_spectra does, using its default central ROI on
    frame `frame_idx` of the scan and `ref_frame_idx` of the phantom). analysis_path is unused.

    Kwargs (all optional):
        speed_of_sound (float): m/s, default 1540.
        bw_cutoff (float): dB threshold for the usable bandwidth, default -10.
        frame_idx (int): 0-based scan frame used for the spectrum estimate, default 0.
        ref_frame_idx (int): 0-based phantom frame, default 0.
        analysis_freq_band (list): [lower, upper] Hz, overrides the spectral estimate.
        center_frequency (float): Hz, overrides the spectral estimate.
        ax_win_size (float): wavelengths, default 15.
        lat_win_size (float): wavelengths, default 15.
        window_thresh, axial_overlap, lateral_overlap (float): defaults 0.95, 0.5, 0.5.
    """
    scan_path = kwargs["scan_path"]
    phantom_path = kwargs["phantom_path"]
    c = float(kwargs.get("speed_of_sound", 1540.0))

    rf, _, _, fs_mhz, _ = read_f2(scan_path, c, frames=int(kwargs.get("frame_idx", 0)))
    ref_rf, _, _, _, _ = read_f2(phantom_path, c, frames=int(kwargs.get("ref_frame_idx", 0)))

    out = RfAnalysisConfig()
    out.sampling_frequency = fs_mhz * 1e6
    out.transducer_freq_band = [0, out.sampling_frequency / 2]
    out.speed_of_sound = c

    if "analysis_freq_band" in kwargs and "center_frequency" in kwargs:
        out.analysis_freq_band = list(kwargs["analysis_freq_band"])
        out.center_frequency = kwargs["center_frequency"]
    else:
        bw = compare_spectra(rf[0], ref_rf[0], out.sampling_frequency,
                             cutoff_db=float(kwargs.get("bw_cutoff", -10.0))).bandwidth
        if not bw.is_valid:
            raise ValueError("Scan and phantom spectra do not overlap at the requested bw_cutoff; "
                             "pass analysis_freq_band and center_frequency explicitly.")
        out.analysis_freq_band = list(kwargs.get("analysis_freq_band", [bw.f_min, bw.f_max]))
        out.center_frequency = kwargs.get("center_frequency", bw.f_center)

    out.ax_win_size = float(kwargs.get("ax_win_size", 15))
    out.ax_win_size_units = WIN_SIZE_UNITS_WAVELENGTH
    out.lat_win_size = float(kwargs.get("lat_win_size", 15))
    out.lat_win_size_units = WIN_SIZE_UNITS_WAVELENGTH
    out.window_thresh = float(kwargs.get("window_thresh", 0.95))
    out.axial_overlap = float(kwargs.get("axial_overlap", 0.5))
    out.lateral_overlap = float(kwargs.get("lateral_overlap", 0.5))

    out.cor_win_size = None
    out.coronal_overlap = None
    return out

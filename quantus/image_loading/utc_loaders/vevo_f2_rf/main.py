import numpy as np
from scipy.signal import hilbert

from ....data_objs.image import UltrasoundRfImage
from ..transforms import scanConvert
from .parser import read_f2, phased_sector_geometry


def _rf_to_bmode(rf: np.ndarray, dynamic_range: float) -> np.ndarray:
    """Log-compressed envelope scaled to 0-255. rf is [..., samples, lines]."""
    with np.errstate(divide="ignore"):
        env_db = 20 * np.log10(np.abs(hilbert(rf, axis=-2)))
    vmax = np.max(env_db[np.isfinite(env_db)])
    env_db = np.clip(np.nan_to_num(env_db, neginf=vmax - dynamic_range), vmax - dynamic_range, vmax)
    return (env_db - (vmax - dynamic_range)) * (255 / dynamic_range)


class EntryClass(UltrasoundRfImage):
    """
    Loader for FUJIFILM VisualSonics Vevo F2 B-mode RF exports.

    Select the <base>.rf.bmode (2D cine) or <base>.rf.3d.bmode (3D sweep) file; the matching
    <base>.rf.xml must sit next to it. Frames of either file are treated as a 2D frame stack
    (for a 3D sweep each frame is one elevational slice), and the frame to analyse is chosen
    at segmentation time, as in readF2.m / MainScript.m.
    """
    extensions = [".bmode"]  # matches both .rf.bmode and .rf.3d.bmode
    spatial_dims = 2
    gui_kwargs = ['speed_of_sound', 'ref_frame_idx', 'dynamic_range']
    cli_kwargs = []
    default_gui_kwarg_vals = ['1540', '0', '50']  # str because parsed from GUI
    default_cli_kwarg_vals = []

    def __init__(self, scan_path: str, phantom_path: str, **kwargs):
        """
        Args:
            scan_path (str): Path to the scan .rf.bmode / .rf.3d.bmode file.
            phantom_path (str): Path to the reference phantom .rf.bmode / .rf.3d.bmode file.
            **kwargs:
                speed_of_sound (float): m/s, used to derive the sampling frequency from the
                    depth axis (default 1540).
                ref_frame_idx (int): 0-based phantom frame used as the reference (default 0).
                dynamic_range (float): B-mode display dynamic range in dB (default 50).
        """
        super().__init__(scan_path, phantom_path)

        c = float(kwargs.get('speed_of_sound', 1540))
        ref_frame_idx = int(kwargs.get('ref_frame_idx', 0))
        dynamic_range = float(kwargs.get('dynamic_range', 50))

        rf, axial, lateral, fs, params = read_f2(scan_path, c, frames=None)
        ref_rf, _, _, ref_fs, ref_params = read_f2(phantom_path, c, frames=ref_frame_idx)
        if not np.isclose(fs, ref_fs, rtol=1e-3):
            raise ValueError(f"Scan ({fs:.3f} MHz) and phantom ({ref_fs:.3f} MHz) sampling "
                             "frequencies differ; acquire the phantom with the same depth settings.")

        self.rf_data = rf                    # [frames, samples, lines]
        self.phantom_rf_data = ref_rf[0]     # [samples, lines]
        self.bmode = _rf_to_bmode(rf, dynamic_range)
        self.axial_res = float(axial[1] - axial[0])  # mm/pix

        # Vevo F2-specific metadata, kept for downstream use (e.g. config loaders / scripts)
        self.sampling_frequency = fs * 1e6  # Hz
        self.speed_of_sound = c
        self.axial = axial
        self.lateral = lateral
        self.f2_params = params
        self.phantom_f2_params = ref_params

        if params.array_type == 'phased':
            tilt, width, r_min, r_max = phased_sector_geometry(params, axial)
            self.tilt = tilt
            self.width = width
            # scanConvert takes depths in metres (same convention as the Clarius loader)
            self.start_depth = r_min / 1000
            self.end_depth = r_max / 1000
            # Angular lines: use the arc length at mid-depth as the lateral pixel size
            self.lateral_res = float(np.deg2rad(width / (rf.shape[2] - 1)) * (r_min + r_max) / 2)

            sc_first, h_cm, w_cm = scanConvert(self.bmode[0], width, tilt,
                                               self.start_depth, self.end_depth, desiredHeight=500)
            self.sc_bmode = np.array([sc_first.scArr] + [
                scanConvert(frame, width, tilt, self.start_depth, self.end_depth,
                            desiredHeight=500)[0].scArr for frame in self.bmode[1:]])
            self.xmap = sc_first.xmap
            self.ymap = sc_first.ymap
            self.sc_axial_res = h_cm * 10 / sc_first.scArr.shape[0]
            self.sc_lateral_res = w_cm * 10 / sc_first.scArr.shape[1]
        else:
            self.lateral_res = float(lateral[1] - lateral[0])  # mm/pix

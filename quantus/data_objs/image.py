import os
import re
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np

PathLike = Union[str, os.PathLike]


def as_path_list(paths: Union[PathLike, Sequence[PathLike]]) -> List[str]:
    """One path or a sequence of paths -> list of str paths (the reference-phantom inputs)."""
    if isinstance(paths, (str, os.PathLike)):
        return [str(paths)]
    out = [str(p) for p in paths]
    if not out:
        raise ValueError("At least one phantom path is required.")
    return out


def _natural_key(path: str):
    """Sort capture_2 before capture_10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path)]


def find_data_files(folder: PathLike, patterns: Sequence[str]) -> List[str]:
    """Data files under `folder` (recursive) matching any glob pattern, in natural order.

    A file inside '<name>_extracted/' is skipped when '<name>.tar' sits next to that folder
    and matched too (the tar would be extracted there again, so it would count twice).
    """
    folder = Path(folder)
    found = {p.resolve() for pattern in patterns for p in folder.rglob(pattern) if p.is_file()}
    tar_dirs = {p.with_name(p.stem + "_extracted") for p in found if p.suffix == ".tar"}
    keep = [p for p in found if not any(d in p.parents for d in tar_dirs)]
    return sorted((str(p) for p in keep), key=_natural_key)


class UltrasoundRfImage:
    """
    Class for ultrasound RF image data.

    phantom_path may be one reference file, a folder, or a list of files/folders. A folder is
    searched recursively for the loader's `folder_patterns` (e.g. Clarius: every capture's
    *rf.raw), so a reference folder holding capture_1..capture_N subfolders can be passed as
    is. Loaders put every reference frame from every file in phantom_rf_frames; spectral
    analysis averages the reference power spectrum over all of them (see
    ParamapAnalysis.get_phantom_window).
    """

    # Glob patterns a loader's data files match, for expanding folders given as phantom_path.
    # None -> this loader does not accept folders.
    folder_patterns: Sequence[str] = None

    def __init__(self, scan_path: str, phantom_path: Union[PathLike, Sequence[PathLike]]):
        # RF data
        self.rf_data: np.ndarray
        self.phantom_rf_data: np.ndarray  # first reference frame, (samples, lines) -- kept for single-frame users
        self.phantom_rf_frames: np.ndarray = None  # every reference frame, (n_frames, samples, lines); 2D loaders only
        self.bmode: np.ndarray
        self.axial_res: float # mm/pix
        self.lateral_res: float # mm/pix
        self.sampling_frequency: float = None # Hz; optional, set by loaders that know it (used by quantus.power_spectrum before a config exists)
        self.scan_name = Path(scan_path).stem
        phantom_inputs = as_path_list(phantom_path)
        self.phantom_paths = self._expand_phantom_inputs(phantom_inputs)  # one entry per reference file
        self.phantom_path = self.phantom_paths[0]  # first reference file (back-compat for single-path users)
        first = Path(phantom_inputs[0])
        self.phantom_name = first.name if first.is_dir() else first.stem
        if len(phantom_inputs) > 1:
            self.phantom_name += f" (+{len(phantom_inputs) - 1} more)"
        elif len(self.phantom_paths) > 1:
            self.phantom_name += f" ({len(self.phantom_paths)} files)"
        self.scan_path = scan_path
        self.spatial_dims: int
        
        # Scan conversion parameters
        self.sc_bmode: np.ndarray = None
        self.xmap: np.ndarray # sc (y,x) --> preSC x
        self.ymap: np.ndarray # sc (y,x) --> preSC y
        self.width: float # deg
        self.tilt: float
        self.start_depth: float # m, from the sector apex (loaders include the probe radius / apex offset)
        self.end_depth: float # m, from the sector apex
        self.sc_axial_res: float # mm
        self.sc_lateral_res: float
        
        # 3D scan parameters
        self.coronal_res: float = None # mm/pix
        self.depth: float # depth in mm
        
        # 3D scan conversion parameters
        self.sc_bmode: np.ndarray = None
        self.coord_map_3d: np.ndarray # maps (z,y,x) in SC coords to (x,y) preSC coord
        self.sc_axial_res: float # mm/pix
        self.sc_lateral_res: float # mm/pix
        self.sc_coronal_res: float # mm/pix
        self.sc_params_3d = None

    def _expand_phantom_inputs(self, inputs: List[str]) -> List[str]:
        """Replace each folder in `inputs` by the data files found in it (folder_patterns)."""
        out = []
        for p in inputs:
            if not os.path.isdir(p):
                out.append(p)
                continue
            if not self.folder_patterns:
                raise ValueError(f"{type(self).__name__} does not accept a folder as phantom input: {p}")
            files = find_data_files(p, self.folder_patterns)
            if not files:
                raise ValueError(f"No reference data files ({', '.join(self.folder_patterns)}) found in folder {p}")
            out.extend(files)
        dupes = {f for f in out if out.count(f) > 1}
        if dupes:
            raise ValueError(f"Reference file(s) given more than once: {', '.join(sorted(dupes))}")
        return out

    def get_phantom_frames(self) -> np.ndarray:
        """All 2D reference frames as (n_frames, samples, lines).

        Falls back to phantom_rf_data for loaders that only set a single frame.
        """
        if self.phantom_rf_frames is not None:
            return np.asarray(self.phantom_rf_frames)
        phantom = np.asarray(self.phantom_rf_data)
        if phantom.ndim != 2:
            raise ValueError(f"get_phantom_frames is for 2D data; phantom_rf_data has shape {phantom.shape}")
        return phantom[None]

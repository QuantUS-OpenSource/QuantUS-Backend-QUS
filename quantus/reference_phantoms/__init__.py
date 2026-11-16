"""Library of calibrated reference phantoms for the reference phantom method.

Each phantom is a folder holding a `phantom.yaml` manifest plus any tabulated data it
points to:

    reference_phantoms/<phantom_name>/
        phantom.yaml       # required
        bsc.txt            # optional, two columns: frequency [MHz], BSC [1/cm-sr]
        attenuation.txt    # optional, two columns: frequency [MHz], attenuation [dB/cm]
        transmission.txt   # optional, two columns: frequency [MHz], membrane amplitude transmission coefficient

Phantoms are discovered from this folder and, additively, from
`$QUANTUS_PLUGIN_DIR/reference_phantoms/` (default `.quantus/plugins/reference_phantoms/`),
which is where calibration data that must not be published belongs. An external phantom
with the same folder name as a built-in one overrides it, with a warning.

Usage in an analysis config (YAML `analysis_kwargs` or `analysis_step(**kwargs)`):

    analysis_kwargs:
      reference_phantom: cirs_rdg_5435   # fills ref_atten*, ref_bsc_path, cirs_path/cirs_mode
      ref_atten: 0.5                     # explicitly given kwargs always win

See README.md in this folder for the manifest schema.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml

from ..plugin_utils import get_external_plugin_root

MANIFEST_NAME = "phantom.yaml"
REFERENCE_PHANTOM_KWARG = "reference_phantom"


@dataclass
class ReferencePhantom:
    name: str                               # folder name, used to reference the phantom
    folder: Path
    display_name: str
    description: str
    # attenuation(f) = atten_coef * f_MHz ** atten_exp + atten_offset   [dB/cm]
    atten_coef: Optional[float] = None
    atten_exp: Optional[float] = None
    atten_offset: Optional[float] = None
    atten_path: Optional[Path] = None
    bsc_path: Optional[Path] = None
    transmission_path: Optional[Path] = None
    membrane_thickness_um: Optional[float] = None

    def _table(self, path: Optional[Path], what: str) -> np.ndarray:
        if path is None:
            raise ValueError(f"Reference phantom '{self.name}' has no {what} data")
        return np.loadtxt(path)

    @staticmethod
    def _interp(freqs_hz, table: np.ndarray) -> np.ndarray:
        # NaN outside the calibrated range rather than silently extrapolating flat
        return np.interp(np.asarray(freqs_hz) / 1e6, table[:, 0], table[:, 1], left=np.nan, right=np.nan)

    def bsc(self, freqs_hz) -> np.ndarray:
        """Backscatter coefficient [1/cm-sr] at the given frequencies [Hz]."""
        return self._interp(freqs_hz, self._table(self.bsc_path, "BSC"))

    def attenuation(self, freqs_hz) -> np.ndarray:
        """Attenuation [dB/cm] at the given frequencies [Hz] (power law if given, else the table)."""
        if self.atten_coef is not None:
            f = np.asarray(freqs_hz) / 1e6
            return self.atten_coef * f ** (self.atten_exp or 1.0) + (self.atten_offset or 0.0)
        return self._interp(freqs_hz, self._table(self.atten_path, "attenuation"))

    def transmission(self, freqs_hz) -> np.ndarray:
        """Membrane amplitude transmission coefficient at the given frequencies [Hz]."""
        return self._interp(freqs_hz, self._table(self.transmission_path, "transmission"))

    def to_analysis_kwargs(self) -> Dict:
        """Analysis kwargs this phantom can supply, using the names analysis methods expect."""
        out = {}
        if self.atten_coef is not None:
            out.update(ref_atten=self.atten_coef, ref_atten_exp=self.atten_exp if self.atten_exp is not None else 1.0,
                       ref_atten_offset=self.atten_offset or 0.0)
        if self.bsc_path is not None:
            out["ref_bsc_path"] = str(self.bsc_path)
        # bsc_transmission_compensation: cirs_mode 2 = tabulated T(f) at cirs_path,
        # cirs_mode 1 = analytic membrane model with cirs_path = thickness in um
        if self.transmission_path is not None:
            out.update(cirs_mode=2, cirs_path=str(self.transmission_path))
        elif self.membrane_thickness_um is not None:
            out.update(cirs_mode=1, cirs_path=str(self.membrane_thickness_um))
        return out


def _resolve(folder: Path, rel: Optional[str]) -> Optional[Path]:
    if rel is None:
        return None
    path = (folder / rel).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{folder / MANIFEST_NAME} references missing file {rel}")
    return path


def load_phantom_folder(folder: Path) -> ReferencePhantom:
    """Parse one <phantom>/phantom.yaml."""
    folder = Path(folder)
    with open(folder / MANIFEST_NAME) as f:
        manifest = yaml.safe_load(f) or {}
    atten = manifest.get("attenuation") or {}
    bsc = manifest.get("bsc") or {}
    trans = manifest.get("transmission") or {}
    return ReferencePhantom(
        name=folder.name,
        folder=folder,
        display_name=manifest.get("name", folder.name),
        description=manifest.get("description", ""),
        atten_coef=atten.get("coef"),
        atten_exp=atten.get("exp"),
        atten_offset=atten.get("offset"),
        atten_path=_resolve(folder, atten.get("file")),
        bsc_path=_resolve(folder, bsc.get("file")),
        transmission_path=_resolve(folder, trans.get("file")),
        membrane_thickness_um=trans.get("thickness_um"),
    )


def _scan(root: Path) -> Dict[str, Path]:
    if not root.is_dir():
        return {}
    return {d.name: d for d in sorted(root.iterdir()) if (d / MANIFEST_NAME).is_file()}


def get_reference_phantoms() -> Dict[str, ReferencePhantom]:
    """All available reference phantoms (built-in + external), keyed by folder name."""
    folders = _scan(Path(__file__).parent)
    external_root = get_external_plugin_root()
    if external_root is not None:
        for name, folder in _scan(external_root / "reference_phantoms").items():
            if name in folders:
                warnings.warn(f"External reference phantom '{name}' overrides the built-in one")
            folders[name] = folder
    return {name: load_phantom_folder(folder) for name, folder in folders.items()}


def get_reference_phantom(name: str) -> ReferencePhantom:
    phantoms = get_reference_phantoms()
    if name not in phantoms:
        raise KeyError(f"Reference phantom '{name}' not found. Available: {', '.join(phantoms) or 'none'}")
    return phantoms[name]


def expand_reference_phantom_kwargs(analysis_kwargs: Dict) -> Dict:
    """If analysis_kwargs names a `reference_phantom`, fill in the kwargs it provides.

    Explicitly passed kwargs take precedence over the phantom's values.
    """
    name = analysis_kwargs.get(REFERENCE_PHANTOM_KWARG)
    if not name:
        return analysis_kwargs
    return {**get_reference_phantom(name).to_analysis_kwargs(), **analysis_kwargs}

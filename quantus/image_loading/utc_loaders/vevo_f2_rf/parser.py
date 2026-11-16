"""Python port of the FUJIFILM VisualSonics Vevo F2 B-mode RF readers.

Ported from the MATLAB sources in the VEVO_F2_QUS repo:
    VEVO_RF_parser/Matlab/VsiParseXml.m     -> parse_vsi_xml
    VEVO_RF_parser/Matlab/VsiBModeRfRead.m  -> read_vsi_bmode_rf
    matlab_src/QUS/readF2.m                 -> read_f2

A Vevo F2 RF export consists of
    <base>.rf.xml                 acquisition / hardware / transducer parameters
    <base>.rf.bmode               2D B-mode RF (a stack of frames), or
    <base>.rf.3d.bmode            3D B-mode RF (a motorised sweep; one frame per slice)

Binary layout of the .rf.bmode / .rf.3d.bmode file (little-endian):
    file header      40 bytes   (uint32 version @0, uint32 numFrames @4, uint32 format @8, ...)
    for each frame:
        frame header 56 bytes
        for each line:
            line header 20 bytes  (uint32 timestamp first)
            numSamples RF samples (int16, int8 or int32 depending on the format word)

Array conventions differ from MATLAB: frames come first, i.e. RF data is returned as
[numFrames, numSamples, numLines], and frame numbers are 0-based throughout.
"""
from __future__ import annotations

import re
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

FILE_HEADER_BYTES = 40
FRAME_HEADER_BYTES = 56
LINE_HEADER_BYTES = 20

# (xml parameter name, output key, type) per mode -- copied from VsiParseXml.m.
_XML_PARAM_LISTS: Dict[str, List[Tuple[str, str, str]]] = {
    "bmode": [
        ("B-Mode/Samples", "BmodeNumSamples", "double"),
        ("B-Mode/Lines", "BmodeNumLines", "double"),
        ("B-Mode/Depth-Offset", "BmodeDepthOffset", "double"),
        ("B-Mode/Depth", "BmodeDepth", "double"),
        ("B-Mode/Width", "BmodeWidth", "double"),
        ("B-Mode/Centre", "BmodeCentre", "double"),
        ("B-Mode/Focal-Zones", "FocalZones", "doubleArray"),
        ("B-Mode/Compounding-Enabled", "CompoundEnabled", "boolean"),
        ("B-Mode/Compounding-Angles", "CompoundAngles", "doubleArray"),
        ("Phased-FOV", "FOV", "double"),
    ],
    "color": [
        ("Color-Mode/Samples", "ColorNumSamples", "double"),
        ("Color-Mode/Lines", "ColorNumLines", "double"),
        ("Color-Mode/Depth-Offset", "ColorDepthOffset", "double"),
        ("Color-Mode/Depth", "ColorDepth", "double"),
        ("Color-Mode/Width", "ColorWidth", "double"),
        ("Color-Mode/Centre", "ColorCentre", "double"),
        ("Color-Mode/TX-PRF", "ColorTxPrf", "double"),
        ("Color-Mode/TX-Frequency", "ColorTxFrequency", "double"),
        ("Sample-Frequency", "ColorRxFrequency", "double"),
        ("Color-Mode/Ensemble-n", "ColorNumEnsemble", "double"),
        ("Color-Mode/Steering-Angle", "ColorSteeringAngle", "double"),
        ("Color-Mode/Group-Size", "ColorGroupSize", "double"),
        ("Color-Mode/Group-Count", "ColorGroupCount", "double"),
        ("Phased-FOV", "FOV", "double"),
    ],
    "power": [
        ("Power-Mode/Samples", "PowerNumSamples", "double"),
        ("Power-Mode/Lines", "PowerNumLines", "double"),
        ("Power-Mode/Depth-Offset", "PowerDepthOffset", "double"),
        ("Power-Mode/Depth", "PowerDepth", "double"),
        ("Power-Mode/Width", "PowerWidth", "double"),
        ("Power-Mode/Centre", "PowerCentre", "double"),
        ("Power-Mode/TX-PRF", "PowerTxPrf", "double"),
        ("Power-Mode/TX-Frequency", "PowerTxFrequency", "double"),
        ("Sample-Frequency", "PowerRxFrequency", "double"),
        ("Power-Mode/Ensemble-n", "PowerNumEnsemble", "double"),
        ("Power-Mode/Steering-Angle", "PowerSteeringAngle", "double"),
        ("Power-Mode/Group-Size", "PowerGroupSize", "double"),
        ("Power-Mode/Group-Count", "PowerGroupCount", "double"),
        ("Phased-FOV", "FOV", "double"),
    ],
    "nlc": [
        ("Nonlinear-Contrast-Mode/Samples", "NLCNumSamples", "double"),
        ("Nonlinear-Contrast-Mode/Lines", "NLCNumLines", "double"),
        ("Nonlinear-Contrast-Mode/Depth-Offset", "NLCDepthOffset", "double"),
        ("Nonlinear-Contrast-Mode/Centre", "NLCCentre", "double"),
        ("Nonlinear-Contrast-Mode/Depth", "NLCDepth", "double"),
        ("Nonlinear-Contrast-Mode/Width", "NLCWidth", "double"),
        ("Nonlinear-Contrast-Mode/Steering-Angle", "NLCSteeringAngle", "double"),
        ("Nonlinear-Contrast-Mode/TX-Power", "NLCTxPower", "double"),
        ("Nonlinear-Contrast-Mode/TX-Frequency", "NLCTxFrequency", "double"),
        ("Sample-Frequency", "NLCRxFrequency", "double"),
    ],
    "3d": [
        ("3D-Scan-Distance", "ScanDistance", "double"),
        ("3D-Step-Size", "ScanStepSize", "double"),
    ],
    "hw": [
        ("Time-Stamp-Clock", "TimestampClock", "double"),
        ("Number-Elements", "NumElements", "double"),
        ("Sample-Frequency", "SampleFreq", "double"),
    ],
    "sw": [
        ("Sound-Speed-Tissue", "SoSTissue", "double"),
    ],
    "transducer": [
        ("Class", "ArrayType", "char"),
        ("Number-Elements", "ArrayElementCount", "double"),
        ("Element-Pitch", "ArrayPitch", "double"),
        ("Array-Radius", "ArrayRadius", "double"),
        ("Lens-Radius", "LensRadius", "double"),
        ("Lens-Thickness", "LensThickness", "double"),
        ("Sound-Speed-Lens", "LensSoS", "double"),
    ],
}

_MODE_ALIASES = {
    ".bmode": "bmode", ".3d.bmode": "bmode",
    ".color": "color", ".3d.color": "color",
    ".power": "power", ".3d.power": "power",
    ".nlc": "nlc", ".3d.nlc": "nlc",
    ".3d": "3d", ".hw": "hw", ".sw": "sw", ".transducer": "transducer",
}

_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _convert_xml_value(raw: str, kind: str):
    if kind == "boolean":
        return raw == "true"
    if kind == "double":
        # Commas mean European decimal formatting; MATLAB replaces them with '.'
        try:
            return float(raw.replace(",", "."))
        except ValueError:
            return float("nan")
    if kind == "doubleArray":
        return np.array([float(x) for x in _NUMBER_RE.findall(raw)])
    return raw  # 'char'


def read_xml_parameters(xml_path: Union[str, Path]) -> Dict[str, str]:
    """Return every <parameter name=... value=...> in a Vevo xml as a raw string dict.

    If a name appears more than once the last occurrence wins, matching VsiParseXml.m.
    """
    xml_path = Path(xml_path)
    if not xml_path.is_file():
        raise FileNotFoundError(f"File ({xml_path}) not found")
    root = ET.parse(xml_path).getroot()
    params = {}
    for node in root.iter("parameter"):
        name = node.get("name")
        if name is not None:
            params[name] = node.get("value", "")
    return params


def parse_vsi_xml(xml_path: Union[str, Path], mode_name: str,
                  raw_params: Optional[Dict[str, str]] = None, warn_missing: bool = True) -> Dict:
    """Port of VsiParseXml.m: read the parameters relevant to one mode.

    Args:
        xml_path: Path to the <base>.rf.xml file.
        mode_name: One of '.bmode', '.3d.bmode', '.color', '.power', '.nlc', '.3d', '.hw',
            '.sw', '.transducer' (with or without the leading dot).
        raw_params: Already-parsed output of read_xml_parameters, to avoid re-reading the xml.
        warn_missing: Warn for each requested parameter not present in the xml.

    Returns:
        dict keyed by the same field names the MATLAB struct used (e.g. 'BmodeNumSamples').
    """
    key = _MODE_ALIASES.get(mode_name, _MODE_ALIASES.get("." + mode_name.lstrip(".")))
    if key is None:
        raise ValueError(f"Mode ({mode_name}) not supported")
    if raw_params is None:
        raw_params = read_xml_parameters(xml_path)

    out = {}
    for xml_name, out_name, kind in _XML_PARAM_LISTS[key]:
        if xml_name in raw_params:
            out[out_name] = _convert_xml_value(raw_params[xml_name], kind)
        elif warn_missing:
            warnings.warn(f"Did not read parameter: {out_name}")
    return out


@dataclass
class VsiBModeRf:
    """Python equivalent of the BMode struct returned by VsiBModeRfRead.m."""
    frame_num: np.ndarray            # [numFramesRead] 0-based frame indices
    timestamp: np.ndarray            # [numFramesRead, numLines] seconds (raw / TimestampClock)
    data: np.ndarray                 # [numFramesRead, numSamples, numLines] RF samples
    width: np.ndarray                # [numLines] lateral axis (mm)
    depth: np.ndarray                # [numSamples] depth axis (mm)
    array_type: str                  # 'linear' or 'phased'
    num_elem: float
    elem_pitch: float                # mm
    lens_thickness: float            # mm
    angle_width: float               # B-Mode/Width (mm, relative to aperture for phased arrays)
    angle_center: float              # B-Mode/Centre (mm)
    fov: float                       # phased arrays only (deg); 0 otherwise
    num_frames_in_file: int
    file_version: int
    bmode_params: Dict = field(default_factory=dict)
    hw_params: Dict = field(default_factory=dict)
    transducer_params: Dict = field(default_factory=dict)


def _rf_sample_dtype(file_format: int) -> np.dtype:
    rf_size = (file_format & 0xF0) >> 4
    if rf_size == 1:
        return np.dtype("<i2")  # MATLAB 'short'
    if rf_size == 2:
        # MATLAB reads this as 'char' (8-bit). RF samples are signed, so read as int8.
        return np.dtype("i1")
    return np.dtype("<i4")


def _resolve_frames(frames: Union[int, Sequence[int], None], num_frames: int) -> np.ndarray:
    if frames is None:
        return np.arange(num_frames)
    frame_arr = np.atleast_1d(np.asarray(frames, dtype=int))
    if np.any(frame_arr == -1):
        if frame_arr.size > 1:
            warnings.warn("Frame list includes -1. Loading all data.")
        return np.arange(num_frames)
    bad = frame_arr[(frame_arr < 0) | (frame_arr >= num_frames)]
    if bad.size:
        raise IndexError(f"Invalid frame number(s) {bad.tolist()} (file has {num_frames} frames, 0-based)")
    return frame_arr


def read_vsi_bmode_rf(base_folder: Union[str, Path], base_filename: str,
                      frames: Union[int, Sequence[int], None] = None,
                      mode_name: str = ".bmode") -> VsiBModeRf:
    """Port of VsiBModeRfRead.m.

    Args:
        base_folder: Folder containing the data.
        base_filename: File name without the '.rf...' extensions.
        frames: 0-based frame index / indices to load; None or -1 loads every frame.
        mode_name: '.bmode' or '.3d.bmode'.
    """
    if mode_name not in (".bmode", ".3d.bmode"):
        raise ValueError(f"mode_name must be '.bmode' or '.3d.bmode', got {mode_name!r}")
    base_folder = Path(base_folder) if base_folder else Path(".")
    xml_path = base_folder / f"{base_filename}.rf.xml"
    raw_params = read_xml_parameters(xml_path)
    param = parse_vsi_xml(xml_path, ".bmode", raw_params, warn_missing=False)
    param_hw = parse_vsi_xml(xml_path, ".hw", raw_params, warn_missing=False)
    param_txr = parse_vsi_xml(xml_path, ".transducer", raw_params, warn_missing=False)
    for req in ("BmodeNumSamples", "BmodeNumLines", "BmodeDepthOffset", "BmodeDepth",
                "BmodeWidth", "BmodeCentre"):
        if req not in param:
            raise KeyError(f"Required parameter {req} missing from {xml_path}")

    fname = base_folder / f"{base_filename}.rf{mode_name}"
    if not fname.is_file():
        raise FileNotFoundError(f"File ({fname}) not found")

    header = np.fromfile(fname, dtype="<u4", count=3)
    if header.size < 3:
        raise ValueError(f"Failed to read file {fname}")
    version, num_frames, file_format = (int(v) for v in header)
    if version < 10:  # 10 or greater is from the F2 system
        raise ValueError("Invalid file version.")

    frame_list = _resolve_frames(frames, num_frames)
    sample_dtype = _rf_sample_dtype(file_format)

    num_samples = int(param["BmodeNumSamples"])
    num_lines = int(param["BmodeNumLines"])
    depth_offset = param["BmodeDepthOffset"]  # mm
    depth = param["BmodeDepth"]               # mm
    width = param["BmodeWidth"]               # mm
    center = param["BmodeCentre"]             # mm

    line_dtype = np.dtype([
        ("timestamp", "<u4"),
        ("line_header", "V", LINE_HEADER_BYTES - 4),
        ("rf", sample_dtype, (num_samples,)),
    ])
    frame_dtype = np.dtype([
        ("frame_header", "V", FRAME_HEADER_BYTES),
        ("lines", line_dtype, (num_lines,)),
    ])
    available = (fname.stat().st_size - FILE_HEADER_BYTES) // frame_dtype.itemsize
    if available < num_frames:
        raise ValueError(f"{fname} holds {available} complete frames but its header claims {num_frames}")

    frames_mm = np.memmap(fname, dtype=frame_dtype, mode="r", offset=FILE_HEADER_BYTES, shape=(num_frames,))
    selected = frames_mm[frame_list]
    # lines['rf'] is [frames, lines, samples] -> [frames, samples, lines]
    data = np.ascontiguousarray(np.transpose(selected["lines"]["rf"], (0, 2, 1)))
    timestamp = selected["lines"]["timestamp"].astype(np.float64)
    del frames_mm

    ts_clock = param_hw.get("TimestampClock")
    if ts_clock:
        timestamp = timestamp / ts_clock
    else:
        warnings.warn("Time-Stamp-Clock missing from xml; timestamps left in raw clock ticks.")

    array_type = param_txr.get("ArrayType", "linear")
    fov = param.get("FOV", 0.0) if array_type == "phased" else 0.0

    return VsiBModeRf(
        frame_num=frame_list,
        timestamp=timestamp,
        data=data,
        depth=np.linspace(depth_offset, depth, num_samples),
        width=np.linspace(0, width, num_lines) - width / 2 + center,
        array_type=array_type,
        num_elem=param_txr.get("ArrayElementCount", float("nan")),
        elem_pitch=param_txr.get("ArrayPitch", float("nan")),
        lens_thickness=param_txr.get("LensThickness", float("nan")),
        angle_width=width,
        angle_center=center,
        fov=fov,
        num_frames_in_file=num_frames,
        file_version=version,
        bmode_params=param,
        hw_params=param_hw,
        transducer_params=param_txr,
    )


def split_f2_path(filename: Union[str, Path]) -> Tuple[Path, str]:
    """Split any file of a Vevo F2 export into (folder, base name without '.rf...')."""
    filename = Path(filename)
    name = filename.name
    idx = name.find(".rf")
    base = name[:idx] if idx >= 0 else name
    return filename.parent, base


def detect_bmode_mode(base_folder: Path, base_filename: str, filename: Union[str, Path, None] = None) -> str:
    """Pick '.bmode' or '.3d.bmode' for a dataset.

    If the user selected a specific .rf.bmode / .rf.3d.bmode file, honour that. Otherwise
    prefer the 3D file when both exist (as readF2.m does).
    """
    if filename is not None:
        name = Path(filename).name
        if name.endswith(".rf.3d.bmode"):
            return ".3d.bmode"
        if name.endswith(".rf.bmode"):
            return ".bmode"
    if (base_folder / f"{base_filename}.rf.3d.bmode").is_file():
        return ".3d.bmode"
    if (base_folder / f"{base_filename}.rf.bmode").is_file():
        return ".bmode"
    raise FileNotFoundError(
        f"Neither .rf.bmode nor .rf.3d.bmode found for base: {base_folder / base_filename}")


@dataclass
class F2Parameters:
    """Equivalent of the `parameters` struct returned by readF2.m (plus the F2 sub-struct)."""
    y_mult: float
    y_zero: float
    x_incr: float          # sample period (s)
    num_points: int        # samples per line
    scan_num1: int         # lines
    scan_num2: int         # frames
    x_zero: float          # depth offset expressed as two-way time (s)
    step1: float           # lateral line spacing (um)
    step2: float
    c: float               # speed of sound (m/s)
    array_type: str
    num_elem: float
    elem_pitch: float
    lens_thickness: float
    angle_width: float
    angle_center: float
    fov: float
    timestamps: np.ndarray
    frame_num: np.ndarray
    mode_name: str
    depth_offset: float    # mm
    depth: float           # mm
    xml_sample_freq: Optional[float]  # 'Sample-Frequency' from the xml, as written there


def read_f2(filename: Union[str, Path], c: float = 1540.0,
            frames: Union[int, Sequence[int], None] = 0
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, F2Parameters]:
    """Port of readF2.m.

    Args:
        filename: Any file of the dataset (<base>.rf.bmode, <base>.rf.3d.bmode, <base>.rf.xml).
        c: Speed of sound (m/s) used to convert the depth axis into a sampling frequency.
        frames: 0-based frame index / indices, or None / -1 for all frames (MATLAB default
            was frame 1, i.e. index 0 here).

    Returns:
        rf_data: [numFrames, numSamples, numLines] float64. Unlike readF2.m, a single frame is
            NOT squeezed, so the shape is always 3-D.
        axial: [numSamples] depth (mm)
        lateral: [numLines] lateral position (mm)
        fs: sampling frequency (MHz), derived from the depth sample spacing and c
        parameters: F2Parameters
    """
    base_folder, base_filename = split_f2_path(filename)
    mode_name = detect_bmode_mode(base_folder, base_filename, filename)
    bmode = read_vsi_bmode_rf(base_folder, base_filename, frames, mode_name)

    rf_data = bmode.data.astype(np.float64)
    num_frames, num_samples, num_lines = rf_data.shape
    axial, lateral = bmode.depth, bmode.width

    # dz = c / (2 fs)  =>  fs = c / (2 dz);  c [m/s]*1e-3 = [mm/us]  ->  fs in MHz
    depth_offset = bmode.bmode_params["BmodeDepthOffset"]
    depth = bmode.bmode_params["BmodeDepth"]
    dz = (depth - depth_offset) / (num_samples - 1)  # mm
    fs = (c * 1e-3) / (2 * dz)

    params = F2Parameters(
        y_mult=1, y_zero=0,
        x_incr=1 / (fs * 1e6),
        num_points=num_samples, scan_num1=num_lines, scan_num2=num_frames,
        x_zero=2 * depth_offset * 1e-3 / c,
        step1=(lateral[-1] - lateral[0]) / (num_lines - 1) * 1e3,
        step2=1000, c=c,
        array_type=bmode.array_type, num_elem=bmode.num_elem, elem_pitch=bmode.elem_pitch,
        lens_thickness=bmode.lens_thickness, angle_width=bmode.angle_width,
        angle_center=bmode.angle_center, fov=bmode.fov,
        timestamps=bmode.timestamp, frame_num=bmode.frame_num, mode_name=mode_name,
        depth_offset=depth_offset, depth=depth,
        xml_sample_freq=bmode.hw_params.get("SampleFreq"),
    )
    return rf_data, axial, lateral, fs, params


def phased_sector_geometry(params: Union[F2Parameters, VsiBModeRf], axial_mm: np.ndarray):
    """Sector geometry for phased arrays, from VsiBModeRfShow.m.

    Returns:
        (tilt_deg, width_deg, r_min_mm, r_max_mm) in the convention of
        transforms.scanConvert (start angle = 270 + tilt - width/2).
    """
    aperture = params.num_elem * params.elem_pitch
    angle_offset = (params.angle_center / aperture) * params.fov
    angle_width = (params.angle_width / aperture) * params.fov
    apex_offset = params.lens_thickness + (aperture / 2) / np.tan(np.deg2rad(params.fov / 2))
    return -angle_offset, angle_width, apex_offset + np.min(axial_mm), apex_offset + np.max(axial_mm)

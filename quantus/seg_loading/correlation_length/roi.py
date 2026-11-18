"""Interactive first-pass ROI support (Python side of DrawROIWithBand.m).

A box dragged on the B-mode (Cartesian mm) becomes a block of RF samples x lines ("sector" ROI:
an annular sector on curvilinear probes, a rectangle on linear ones), exactly as
DrawROIWithBand's UpdateOutline; roi_band_check (first_pass.py) gives its band. The accepted ROI
is saved in the QuantUS UI's ROI pickle format, so the standard `pkl_roi` segmentation loader
turns it into the segmentation (scan-converted outline for sector images, RF line/sample outline
otherwise, as the manual ROI drawing does).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .first_pass import FirstPassIntersect, ROIBand, roi_band_check


@dataclass
class SectorROI:
    """RF sample rows r1..r2 x lines l1..l2 (0-based, inclusive) and its Cartesian outline (mm)."""
    rows: tuple
    lines: tuple
    x_mm: np.ndarray
    z_mm: np.ndarray
    depth_range_mm: tuple
    band: Optional[ROIBand] = field(default=None, repr=False)
    band_error: Optional[str] = None

    @property
    def mask_shape_ok(self) -> bool:
        return self.rows[1] >= self.rows[0] and self.lines[1] >= self.lines[0]

    def mask(self, n_samples: int, n_lines: int) -> np.ndarray:
        m = np.zeros((n_samples, n_lines), dtype=bool)
        m[self.rows[0]:self.rows[1] + 1, self.lines[0]:self.lines[1] + 1] = True
        return m


def sector_roi_from_box(axial_mm: np.ndarray, lateral_mm: np.ndarray, x0: float, x1: float, z0: float,
                        z1: float) -> SectorROI:
    """Box (lateral x0..x1, depth z0..z1 mm) -> block of RF rows x lines (DrawROIWithBand 'sector').

    Rows: the samples on the centre line nearest the box top / bottom. Lines: those whose lateral
    position at the middle row lies inside the box (the nearest one if none). Outline: along the
    top row, down the last line, back along the bottom row, up the first line.
    """
    axial = np.asarray(axial_mm, dtype=float)
    lateral = np.asarray(lateral_mm, dtype=float)
    n_lines = axial.shape[1]
    mid = int(np.sign(n_lines / 2) * np.floor(abs(n_lines / 2) + 0.5)) - 1
    centre = axial[:, mid]
    za, zb = sorted((z0, z1))
    xa, xb = sorted((x0, x1))
    r1, r2 = int(np.argmin(np.abs(centre - za))), int(np.argmin(np.abs(centre - zb)))
    if r2 < r1:
        r1, r2 = r2, r1
    rm = int(np.floor((r1 + r2) / 2 + 0.5))
    inside = np.nonzero((lateral[rm] >= xa) & (lateral[rm] <= xb))[0]
    if inside.size == 0:
        inside = np.array([int(np.argmin(np.abs(lateral[rm] - (xa + xb) / 2)))])
    l1, l2 = int(inside.min()), int(inside.max())
    return sector_roi_from_rows_lines(axial, lateral, (r1, r2), (l1, l2))


def sector_roi_from_rows_lines(axial_mm: np.ndarray, lateral_mm: np.ndarray, rows, lines) -> SectorROI:
    """SectorROI for RF rows r1..r2 x lines l1..l2 (0-based, inclusive), e.g. read back from an ROI
    pickle: outline along the top row, down the last line, back along the bottom row, up the first.
    depth_range_mm: rows r1 / r2 on the centre line."""
    axial = np.asarray(axial_mm, dtype=float)
    lateral = np.asarray(lateral_mm, dtype=float)
    n_rows, n_lines = axial.shape
    (r1, r2), (l1, l2) = sorted(int(v) for v in rows), sorted(int(v) for v in lines)
    if not (0 <= r1 <= r2 < n_rows and 0 <= l1 <= l2 < n_lines):
        raise ValueError(f"ROI rows {r1}-{r2} / lines {l1}-{l2} outside the {n_rows} x {n_lines} RF frame")
    mid = int(np.sign(n_lines / 2) * np.floor(abs(n_lines / 2) + 0.5)) - 1
    cols = np.arange(l1, l2 + 1)
    rr = np.arange(r1, r2 + 1)
    xo = np.concatenate([lateral[r1, cols], lateral[rr, l2], lateral[r2, cols[::-1]], lateral[rr[::-1], l1]])
    zo = np.concatenate([axial[r1, cols], axial[rr, l2], axial[r2, cols[::-1]], axial[rr[::-1], l1]])
    return SectorROI(rows=(r1, r2), lines=(l1, l2), x_mm=xo, z_mm=zo,
                     depth_range_mm=(float(axial[r1, mid]), float(axial[r2, mid])))


def check_sector_roi(intersect: FirstPassIntersect, roi: SectorROI) -> SectorROI:
    """Fill roi.band with roi_band_check on its outline (roi.band_error if no blocks inside)."""
    try:
        roi.band, roi.band_error = roi_band_check(intersect, roi.x_mm, roi.z_mm), None
    except ValueError as e:
        roi.band, roi.band_error = None, str(e)
    return roi


def rf_outline_points(roi: SectorROI, max_per_side: int = 100) -> tuple:
    """Closed outline of the ROI in RF coordinates (lines, samples): top row, last line, bottom
    row, first line, each side with at most max_per_side points, corners included."""
    (r1, r2), (l1, l2) = roi.rows, roi.lines

    def side(a, b):
        n = min(abs(b - a) + 1, max_per_side)
        return np.linspace(a, b, n)

    top, right = side(l1, l2), side(r1, r2)
    x = np.concatenate([top, np.full(right.size, l2), top[::-1], np.full(right.size, l1)])
    y = np.concatenate([np.full(top.size, r1), right, np.full(top.size, r2), right[::-1]])
    return x, y


def roi_spline_for_pkl(image_data, roi: SectorROI, max_points: int = 600) -> tuple:
    """Spline (X, Y) in the coordinates the `pkl_roi` loader expects for this image.

    No scan conversion: RF (line, sample). Scan-converted image (sc_bmode set): the ROI's
    Cartesian outline (mm) mapped to scan-converted pixels with the affine map fitted from
    xmap / ymap and the RF sample positions (smooth outline; pkl_roi's xmap / ymap lookup lands on
    the ROI's lines and samples). Falls back to the nearest scan-converted pixel of each RF
    outline point if the image geometry is not available or the fit is poor.
    """
    x, y = rf_outline_points(roi)
    if getattr(image_data, "sc_bmode", None) is None:
        return x, y
    mapped = _sc_pixels_from_mm(image_data, roi.x_mm, roi.z_mm, max_points)
    if mapped is not None:
        return mapped
    return _nearest_sc_pixels(image_data, x, y)


def _sc_valid_pixels(image_data):
    xmap = np.asarray(image_data.xmap, dtype=float)
    ymap = np.asarray(image_data.ymap, dtype=float)
    sc = np.asarray(image_data.sc_bmode)
    sc = sc[0] if sc.ndim == 3 else sc
    valid = np.isfinite(xmap) & np.isfinite(ymap)
    if sc.shape == xmap.shape:
        valid &= sc > 0  # inside the sector (outside is zero)
    iy, ix = np.nonzero(valid)
    return xmap, ymap, iy, ix


def _sc_pixels_from_mm(image_data, x_mm, z_mm, max_points: int, max_residual_px: float = 3.0):
    """Affine map (lateral, axial mm) -> scan-converted (x, y) pixel fitted on the sector pixels;
    None if it can't be fitted or does not fit to within max_residual_px."""
    try:
        from .correlation import positions_from_image

        xmap, ymap, iy, ix = _sc_valid_pixels(image_data)
        if iy.size < 10:
            return None
        if iy.size > 20000:
            pick = np.random.default_rng(0).choice(iy.size, 20000, replace=False)
            iy, ix = iy[pick], ix[pick]
        rf = np.asarray(image_data.rf_data)
        n_samples, n_lines = rf.shape[-2:]
        axial, lateral = positions_from_image(image_data, n_samples, n_lines)
        lines = np.clip(np.round(xmap[iy, ix]).astype(int), 0, n_lines - 1)
        samples = np.clip(np.round(ymap[iy, ix]).astype(int), 0, n_samples - 1)
        a = np.column_stack([lateral[samples, lines], axial[samples, lines], np.ones(iy.size)])
        cx, *_ = np.linalg.lstsq(a, ix.astype(float), rcond=None)
        cy, *_ = np.linalg.lstsq(a, iy.astype(float), rcond=None)
        if max(np.percentile(np.abs(a @ cx - ix), 99), np.percentile(np.abs(a @ cy - iy), 99)) > max_residual_px:
            return None
    except Exception:
        return None
    x_mm, z_mm = np.asarray(x_mm, float), np.asarray(z_mm, float)
    if x_mm.size > max_points:
        keep = np.unique(np.linspace(0, x_mm.size - 1, max_points).round().astype(int))
        x_mm, z_mm = x_mm[keep], z_mm[keep]
    b = np.column_stack([x_mm, z_mm, np.ones(x_mm.size)])
    h, w = xmap.shape
    # +0.5: pkl_roi looks the pixel up with int() (floor); this makes it the nearest pixel
    return np.clip(b @ cx + 0.5, 0, w - 1), np.clip(b @ cy + 0.5, 0, h - 1)


def _nearest_sc_pixels(image_data, x, y) -> tuple:
    """For each RF outline point (line x, sample y), the scan-converted pixel whose (xmap, ymap)
    is nearest."""
    from scipy.spatial import cKDTree

    xmap, ymap, iy, ix = _sc_valid_pixels(image_data)
    if iy.size == 0:  # e.g. sc_bmode not in positive units: use every mapped pixel
        iy, ix = np.nonzero(np.isfinite(xmap) & np.isfinite(ymap))
    # one scan-converted pixel spans ~1 line laterally but many samples axially: compare in units
    # of the sc pixel footprint so neither direction dominates the nearest-neighbour search
    s = _samples_per_sc_pixel(image_data)
    tree = cKDTree(np.column_stack([xmap[iy, ix], ymap[iy, ix] / s]))
    _, idx = tree.query(np.column_stack([x, y / s]))
    return ix[idx].astype(float), iy[idx].astype(float)


def _samples_per_sc_pixel(image_data) -> float:
    """RF samples per scan-converted pixel along the beam (sc axial pixel / RF sample spacing)."""
    ax_res = float(getattr(image_data, "axial_res", 0) or 0)
    sc_res = float(getattr(image_data, "sc_axial_res", 0) or 0)
    return max(1.0, sc_res / ax_res) if ax_res > 0 and sc_res > 0 else 1.0


def save_roi_pkl(path, image_data, roi: SectorROI, frame: int = 0, extra: Optional[dict] = None) -> Path:
    """Save the ROI in the QuantUS UI ROI pickle format (Spline X / Y, Scan Name, Phantom Name,
    Frame) plus the first-pass details, for the `pkl_roi` loader."""
    sx, sy = roi_spline_for_pkl(image_data, roi)
    info = {
        "Spline X": tuple(sx), "Spline Y": tuple(sy),
        "Scan Name": getattr(image_data, "scan_name", ""), "Phantom Name": getattr(image_data, "phantom_name", ""),
        "Frame": int(frame),
        "Source": "QuantUS first-pass ROI (DrawROIWithBand)",
        "Created": datetime.now().isoformat(timespec="seconds"),
        "RF Rows": tuple(int(v) for v in roi.rows), "RF Lines": tuple(int(v) for v in roi.lines),
        "Depth Range mm": tuple(float(v) for v in roi.depth_range_mm),
    }
    if roi.band is not None:
        info.update({"Band MHz": (roi.band.f_low_roi, roi.band.f_high_roi),
                     "All Rows Usable": roi.band.all_rows_usable})
    if extra:
        info.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(info, f)
    return path


def load_roi_pkl_info(path) -> Optional[dict]:
    """The first-pass details saved with a Power Guided ROI (save_roi_pkl): RF Rows, RF Lines,
    Depth Range mm, Band MHz, ... None if the file is not such an ROI (e.g. a hand-drawn ROI)."""
    try:
        with open(path, "rb") as f:
            info = pickle.load(f)
    except Exception:
        return None
    if not isinstance(info, dict) or not all(k in info for k in ("RF Rows", "RF Lines", "Depth Range mm")):
        return None
    return info

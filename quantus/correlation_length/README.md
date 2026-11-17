# Correlation lengths

Axial and lateral speckle correlation lengths vs. depth, used to choose QUS window sizes
(e.g. ~10 axial correlation lengths per window). Usually run on the **reference phantom**
right after image loading, before the analysis config. Port of
`CalcAxLatCorrelationCoeffCurves.m` (S1Study, 2026-11-16 version: window centres from
`AxialPosition`). Checked against it on the same `RFData.mat` frames (C3, L15, L20, 2-3 frames,
38-68 depths, and `[]` depth inputs): identical depths and NaN pattern; the lengths are
0.001-0.002 shift shorter, from the exact threshold crossing (see "Changed from MATLAB").
Before that fix the two agreed to <= 9e-16 mm.

```python
from quantus.entrypoints import scan_loading_step
from quantus.correlation_length import correlation_lengths_from_image, plot_correlation_lengths

image = scan_loading_step("clarius_rf", scan_path, reference_folder)  # every capture in the folder
res = correlation_lengths_from_image(image, source="phantom",
                                     window_length_mm=10, window_width_mm=30,  # measurement window
                                     ax_start_mm=8, ax_end_mm=143, n_ax_locations=68,
                                     threshold=0.5,
                                     ax_window_factor=10, lat_window_factor=5)  # recommended window multipliers
plot_correlation_lengths(res)        # lengths in samples / A-lines, recommended whole-number windows
res.to_table()                       # per depth: samples / A-lines and mm
res.recommended_ax_window_samples, res.recommended_lat_window_lines   # whole numbers, per depth
print(res.describe_window())         # ONE window for the whole checked depth range (largest per-depth value)
res.recommended_window()             # same as a dict
res.lat_window_factor = 4            # factors can be changed afterwards; recommendations recompute

# 2D: the same measurement on a grid of window positions across the whole image
from quantus.correlation_length import correlation_length_map_from_image, plot_correlation_length_map
cmap = correlation_length_map_from_image(image, window_length_mm=10, window_width_mm=15)  # lat_step_lines=None: every half window
print(cmap.describe_window())                                             # largest anywhere in the image
print(cmap.describe_window(depth_range_mm=(10, 95), lateral_range_mm=(-40, 40)))  # ...or in a region
plot_correlation_length_map(cmap, depth_range_mm=(10, 95), lateral_range_mm=(-40, 40))
cmap.to_table()                                                           # one row per measured window
```

The whole-image maximum is set by the single worst window, and walls, near-field
reverberation or point targets give much longer correlation lengths than speckle (e.g. C3
phantom: 660 samples axially at a side wall vs ~77 in the speckle). Restrict the region to the
speckle / ROI you will analyse.

- `correlation_lengths(rf_frames, axial_position_mm, lateral_position_mm, ...)` is the
  scanner-independent core: RF frames `(frames, samples, lines)` plus the physical position
  (mm) of every sample, as in the MATLAB `RFData.mat`.
- `positions_from_image` builds those positions from a loaded image: sector/curvilinear
  geometry from `width` (deg), `tilt` and `start_depth` (apex-to-first-sample, m); otherwise
  linear from `axial`/`lateral` or `axial_res`/`lateral_res`.
- Outputs per depth and frame: full axial correlation length (`ax_corr_mm`, the threshold
  crossing x 2 -- pulse-echo, so the axial length is the full width, as in the literature) and
  lateral correlation width (`lat_corr_mm`, the threshold crossing); means
  and SDs across frames, in mm and in samples (`ax_mean_samples`) / A-lines (`lat_mean_lines`,
  using the line spacing at that depth).
- Recommended window = `ax_window_factor` (default 10) x axial mean and `lat_window_factor`
  (default 5, the MATLAB `0.5*10`) x lateral mean, **rounded up to a whole number** of samples /
  A-lines (`recommended_ax_window_samples`, `recommended_lat_window_lines`); the `_mm` versions
  are those whole numbers x the spacing. NaN where the length is undefined.

## Window centres (depths)
- `ax_start_mm` / `ax_end_mm` snap to the nearest sample on the centre line, using the axial
  positions (so any RF start delay is included), and are kept far enough inside the image for
  a full window. `None` means the first / last full window in the image.
- `n_ax_locations` sets roughly how many depths (spacing = range / n, at least 1 sample);
  `None` puts a window centre every half window.
- Reported depths are the axial positions of the window centres.

## Method (per frame, per depth)
1. Linear-detrend each A-line, Hilbert envelope.
2. Window centred on the middle A-line (`correlation_lengths`) or on each grid position
   (`correlation_length_map`): `window_length_mm` deep, `window_width_mm` wide (odd numbers of
   samples and lines). Spacings are straight-line distances between neighbouring samples
   (along the line; to the neighbouring line at the window's depth), so linear, curvilinear
   and tilted geometry are all handled.
3. Detrend each A-line of the window envelope, add its mean back, square (intensity).
4. Lateral curve: correlation coefficient between A-lines c and c+shift (over depth), mean
   over c. Axial curve: between rows r and r+shift (over A-lines), mean over r.
5. The first shift below `threshold`, interpolated linearly between the two neighbouring
   shifts, gives the length.
   If a curve never drops below `threshold`, that length is NaN (excluded from the means).

## Changed from MATLAB
- Axial curve never below the threshold: MATLAB returned `max(curve) * spacing / 1000`
  (a correlation *value* times spacing, not a distance); the port now returns NaN, as it
  already did for the lateral curve.
- Threshold crossing: solved exactly on the straight segment between the two shifts around it.
  MATLAB searches a 0.001-shift grid and also reports one step late (1-based index), so its
  lengths are 0.001-0.002 shift longer.
- Spacings: straight-line distances between neighbouring samples. MATLAB uses differences of
  the axial / lateral coordinates, which underestimate the line spacing by ~cos(tilt) for a
  tilted sector (identical for linear and untilted probes, to ~1e-5).
- 2D map (`correlation_length_map`): new; the MATLAB measures the centre line only.

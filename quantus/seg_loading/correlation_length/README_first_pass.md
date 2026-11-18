# First-pass power check

Where in the image is there enough signal for spectral analysis? **This is a quick power check
only, used to choose the ROI: the spectra used for parameter estimation are recomputed later
with the final window settings** (the first pass uses a simple Hanning-windowed FFT averaged over
blocks, not the analysis spectral estimator). Port of the MATLAB first-pass
functions (`S1Study/Code/Functions`, versions of 2026-11-12/13; the `Prostate/Code` copies of
2026-11-16 compute the same and only generalise names / plots). Lives here for now; to be
reconciled with `quantus/seg_loading/power_spectrum` at merge.

| MATLAB | Python (`first_pass.py`) |
|---|---|
| `CalcFirstPassPowerSpectra` | `calc_first_pass` / `calc_first_pass_from_image` |
| `FindBWEdges` | `find_bw_edges` |
| `IntersectFirstPass` | `intersect_first_pass` |
| `ROIBandCheck` | `roi_band_check` (band of a given ROI) |

The automatic ROI choice (`AutoROIFromFirstPass`) is deliberately not included.

## Pipeline: correlation-length windows -> first pass
```python
from quantus.seg_loading.correlation_length import windows_and_first_pass, plot_first_pass
res = windows_and_first_pass(image, preset="C3")   # image = scan_loading_step(...): sample + reference
print(res.summary())
res.ax_window_mm, res.lat_window_mm                # from the reference correlation lengths
res.intersect.usable_both_2d                       # blocks usable in sample AND reference
plot_first_pass(res, image)
```
1. `correlation_lengths_from_image` on the reference phantom gives the recommended window (mm).
2. `calc_first_pass` on the reference (all captures) and the sample uses it as the axial window and
   lateral block width (depth step 0.2 x the axial window by default), with the image's true
   sampling frequency (`use_image_sampling_frequency=False` for the MATLAB formula).
3. `intersect_first_pass`: blocks usable in both.

Method: Hanning-windowed (MATLAB `hanning`, no zero end points, unit energy) spectra at every
depth, FFT length = window length, averaged over frames and over all lines (depth profile) or
over lateral blocks of a fixed number of lines with half overlap (2-D map). Band edges = noise
floor + 10 dB, walking out from the peak and stopping at notches; noise floor = mean power of the
flat high-side plateau. Usable = a band exists and peak >= noise floor + 15 dB. Indices are
0-based in Python (`block_centre`, `block_start`, `rows`).

## Signal up to Nyquist: stop
The noise floor comes from the flat out-of-band region above the signal. If the signal band runs
up to Nyquist there is none: `FindBWEdges` falls back to the minimum of the spectrum, which jumps
from depth to depth (Clarius L15, fs 20 MHz: noise floor 21.7 -> 46.5 dB between 18.7 and
21.2 mm, giving depth-banded SNR maps). This is known for L15 with imaging depth past ~25 mm.
`calc_first_pass` checks every depth-profile spectrum: no high-side plateau AND a band running into
the top bin -> `nyquist_limited`. By default it raises `NyquistError` ("do not use this data");
`on_nyquist="warn"` / `"ignore"` instead. On the test data: L15 flagged (reference 25 of 76 depths,
sample 12 of 76); C3 and L20 never.

## Verification against MATLAB (R2026b)
Both ran on the same `RFData.mat` frames (Clarius C3, L15, L20 reference phantom: 5 captures;
sample phantom: 1 capture), exported from the QuantUS loader. Six first-pass cases (whole image
with depth step; depth range with a fixed number of depths; L20 with a 7.5 MHz avoided notch and
peak search above 7.5 MHz) and 168 `FindBWEdges` calls on real spectra with every option path
(plateau / band / legacy noise floor, plateau-to-legacy fallback, notch-limited edges on both
sides, guard, avoid frequencies, search-above, edge bins kept). Every array, band edge, usable map,
block grid, intersection and `ROIBandCheck` output agreed; the largest difference was 4e-14
(floating point). Checked 2026-11-16 (before the Nyquist check, which the MATLAB does not have);
the export / MATLAB / comparison scripts are not in the repo.

## Known issues in the MATLAB (ported unchanged unless noted)
- **Sampling frequency**: `fs = 1.540 / 2 / dz` from `AxialPosition`. When the axial spacing was
  computed with another sound speed the frequency axis is stretched: on the Clarius test data
  15.03 / 20.02 / 60.37 MHz instead of 15 / 20 / 60 MHz (0.2 / 0.1 / 0.6 %). `calc_first_pass`
  keeps the MATLAB formula by default (`sampling_frequency_mhz` overrides it); the pipeline uses
  the image's true sampling frequency.
- **Odd FFT length**: `PS(2:end-1)` is doubled, so the last bin (not Nyquist for an odd length) is
  3 dB low. No effect by default: `FindBWEdges` drops the first and last bins.
- **Block width**: lines per block from the x-difference of neighbouring lines at mid depth (exact
  for linear probes and untilted sectors; slightly short on a tilted sector).
- **Different RF start**: `IntersectFirstPass` assumes sample and reference rows start at the same
  depth; `LiverFirstPassROI`'s `AlignedDepthRange` / `AlignFirstPass` (RF delay offsets) are not
  ported.
- `AutoROIFromFirstPass` (not ported): its no-box error message fails in R2026b ("Formatted
  arguments must be scalar").

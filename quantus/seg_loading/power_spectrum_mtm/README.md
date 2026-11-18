# power_spectrum_mtm — multitaper power spectra for an ROI

Port of the S1Study MATLAB chain (Code/ and Code/Functions/, Nov 2026). This is the step after
the first-pass ROI is accepted: the first pass (correlation_length/first_pass.py) is only a quick
power check, and the spectra are recomputed here for parameter estimation. It is kept apart from
`quantus/seg_loading/power_spectrum` (a collaborator's module) until the merge.

| Step | MATLAB | Python |
|---|---|---|
| Correlation lengths inside the ROI → final window | CalcAxLatCorrelationCoeffCurves | `correlation_lengths(..., line_range=, depth_range_mm=)` |
| NW from the window and the ROI band | DetermineSpectralParameters.m + CheckMTMParameters.m | `determine_spectral_parameters`, `check_mtm_parameters` |
| Window grid over the ROI | ComputePowerSpectra.m + DefinePowerSpectrumLocations.m | `define_power_spectrum_locations` |
| Multitaper spectra (reference mean, sample) | ComputePowerSpectra_MTM_Grid(_Reference), PowerSpectrumMTF, CreateDPSS | `grid_power_spectra`, `power_spectrum_mtf` |
| Band per window, shared by sample and reference | DetermineBandwidth.m, DetermineBW_Reference, DetermineBW_ROI | `roi_bandwidth`, `bw_options` |
| Whole chain | — | `power_spectral_estimation(_from_image)` |

## Rules

- **Correlation window:** reference phantom, windows every half window across the ROI's A-lines
  (centred on its middle line) and lying entirely inside its depth range; 10× axial / 5× lateral.
- **NW:** band = ROI first-pass band rounded inward to 0.1 MHz; N = round(L/dz) made odd;
  NW = κ·B·T (κ = 0.2, T = N/fs) to the nearest 0.5, capped if `avoid_freq` would leak into
  [fLow − W, fHigh + W], flagged if < 1.5; K = floor(2NW − 1); W = NW/T.
- **Grid:** window centres from the ROI depth range rounded outward to 0.1 mm and the ROI's
  lines; overlap ≥ 0.92 axial, ≥ 0.85 lateral. Edge windows reach half a window outside the ROI.
- **Band:** FindBWEdges with Guard = W, no automatic notch detection, AvoidFreq/SearchAbove.
  Window usable if sample and reference bands overlap and both are ≥ 5 dB above their band level.
  Strict band = [max shared fLow, min shared fHigh]; 95/5 band = percentiles (MATLAB `prctile`).

**Not used (for now):** DetermineInclusionPSERs / DetermineAttenuationPER. Every grid window is
evaluated.

**Known difference inside the MATLAB:** NW is chosen for N = round(L/dz) made odd, but the grid
windows are round(L/dz) + 1 samples. When round(L/dz) is odd they differ by one sample (W changes
by < 1%). The port keeps both as the MATLAB does.

## Speed

C3 (123 × 102 windows, 5 reference frames) takes about 2 minutes, mostly the reference spectra.

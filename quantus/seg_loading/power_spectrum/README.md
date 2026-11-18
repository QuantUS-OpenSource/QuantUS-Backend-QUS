# Power spectrum inspection

Scanner-independent check of the sample vs. reference phantom power spectra, meant to be run
**after image loading and before segmentation / analysis config** to choose the analysis band
from the shape of the spectra (port of `showPowerF2.m`, verified against it).

```python
from quantus.entrypoints import scan_loading_step
from quantus.power_spectrum import compare_image_spectra, apply_bandwidth_to_config, plot_spectrum_comparison

image = scan_loading_step("vevo_f2_rf", scan_path, phantom_path)
result = compare_image_spectra(image, frame=0,
                               sample_roi=(r0, r1, c0, c1),     # pre-scan-conversion pixels
                               reference_roi=(r0, r1, c0, c1),  # None -> central default ROI
                               cutoff_db=-10)
plot_spectrum_comparison(result)
apply_bandwidth_to_config(config, result.bandwidth)  # sets analysis_freq_band + center_frequency
```

- ROIs are `(row_start, row_end, col_start, col_end)`, 0-based, exclusive ends, in **pre-scan-conversion** RF indices.
- The analysis band is the intersection of the sample's and the reference's bands, each defined as where the spectrum stays above `cutoff_db` relative to its own peak.
- `compare_image_spectra` uses `image_data.sampling_frequency` (Hz) when the loader sets it, so it can run before a config exists; otherwise pass `sampling_frequency`. `compare_spectra` works on raw 2D RF frames.
- All frequencies are in Hz.

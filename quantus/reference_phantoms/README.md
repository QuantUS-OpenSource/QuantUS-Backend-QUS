# Reference phantoms

Calibration data for reference phantoms used by the reference phantom method: attenuation,
backscatter coefficient (BSC), and membrane transmission coefficient. One folder per phantom:

```
reference_phantoms/<phantom_name>/
├── phantom.yaml        # required manifest
├── bsc.txt             # frequency [MHz], BSC [1/cm-sr]
├── attenuation.txt     # frequency [MHz], attenuation [dB/cm]    (optional if a power law is given)
└── transmission.txt    # frequency [MHz], amplitude transmission coefficient of the membrane
```

`phantom.yaml`:

```yaml
name: CIRS RDG 5435
description: free text (source / calibration date / lot)
attenuation:            # attenuation(f) = coef * f_MHz^exp + offset  [dB/cm], and/or file:
  coef: 0.2089
  exp: 1.4183
  offset: 0.2712
  # file: attenuation.txt
bsc:
  file: bsc.txt
transmission:           # either a measured table ...
  file: transmission.txt
  # ... or a membrane thickness for the analytic model in bsc_transmission_compensation
  # thickness_um: 25
```

## Using a phantom in analysis

Name it in `analysis_kwargs`; it fills `ref_atten`, `ref_atten_exp`, `ref_atten_offset`,
`ref_bsc_path`, and `cirs_mode`/`cirs_path` (transmission). Kwargs you set explicitly win.

```yaml
analysis_kwargs:
  reference_phantom: cirs_rdg_5435
```

From Python: `get_reference_phantoms()`, `get_reference_phantom(name).bsc(freqs_hz)` /
`.attenuation(freqs_hz)` / `.transmission(freqs_hz)` / `.to_analysis_kwargs()`.

## Private calibration data

Calibration data that must not be published goes in
`$QUANTUS_PLUGIN_DIR/reference_phantoms/<phantom_name>/` (default `.quantus/plugins/`, gitignored)
with the same layout. It is discovered alongside the built-in phantoms; a same-named folder
overrides the built-in one.

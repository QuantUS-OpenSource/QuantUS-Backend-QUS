# Analysis plugins

This directly contains all QUS analysis methods used for analysis. The plugin system enables users to extend QuantUS with new QUS methods by adding files to `paramap/analysis_methods` or `bmode/analysis_methods` (window/full_segmentation plugins), or `paramap/analysis_aggregators`/`bmode/analysis_aggregators` (aggregate plugins — see "Three plugin locations" below).

All analysis is currently based around parametric maps via the sliding window technique. Every plugin function is discovered the same way (`discover_marked_functions`, keyed off the `outputs` attribute set by `@output_vars`), and what differs between the three kinds below is purely the `@location(...)` decorator and, correspondingly, when the function runs and what it receives — except `aggregate` functions also live in a separate sibling folder (see "Three plugin locations" below) so they're visually distinct from the window/full_segmentation plugins they aggregate.

## Single-file vs. folder plugins

A plugin can be either a single `.py` file (the common case — most plugins are self-contained) or a folder containing an `__init__.py` (e.g. `homodyned_k_params/__init__.py`), discovered identically either way. Use the folder form when a plugin needs to bundle a sibling data asset (a lookup table, calibration data, etc.) that's a fixed constant for the method itself rather than a per-run/per-scan config value — load it via a path relative to the plugin's own `__file__` (e.g. `Path(__file__).parent / "my_data.mat"`), not a `required_kwargs` path. `homodyned_k_params/` is the reference example: it bundles `RSK_data_501.mat`, a universal precomputed lookup table the estimator needs on every call, so nothing in a run's config has to know about it.

This is a per-plugin choice, orthogonal to which of `analysis_methods/`/`analysis_aggregators/` the plugin lives in — either folder may contain a mix of flat-file and folder plugins. A folder here always means exactly one plugin (never a category of several) — see "Three plugin locations" for why aggregate plugins get their own top-level sibling folder instead of a category subfolder nested inside `analysis_methods/`.

## External / private plugins

Plugins don't have to live in this repo. Set `QUANTUS_PLUGIN_DIR` (defaults to `.quantus/plugins/` at the repo root) to a directory containing an `analysis/` subfolder mirroring this layout (e.g. `$QUANTUS_PLUGIN_DIR/analysis/paramap/analysis_methods/your_method.py`, `$QUANTUS_PLUGIN_DIR/analysis/paramap/analysis_aggregators/your_aggregator.py`, or a whole new `$QUANTUS_PLUGIN_DIR/analysis/your_type/` analysis type), and it's picked up alongside the built-ins — no changes to this repo needed. This is how private/proprietary QUS methods are meant to be distributed (an approved user drops a received plugin file in) and how a future GUI can load plugins dynamically. A plugin here with the same name as a built-in one overrides it (with a warning); everything else stays available.

## Three plugin locations

The `location` decorator determines which of these three shapes a plugin follows. Declaring no `location` at all defaults to `['window', 'full_segmentation']` (the function runs at both). `'aggregate'` is never a default — it must be declared explicitly.

### 1. Window (`@location('window')`, or undeclared)

The common case: called once per sliding-window tile generated over the segmentation.

```python
def METHOD_NAME(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray,
                    window: Window, config: RfAnalysisConfig,
                    image_data: UltrasoundRfImage, **kwargs) -> None:
```

* `scan_rf_window` — the current window's slice of RF data (2D: samples × channels, or 3D: samples × channels × slices). Pre-scan-conversion.
* `phantom_rf_window` — the same spatial window, but from the reference phantom scan.
* `window` — spatial metadata plus `window.results`, a dynamically-populated bag of prior results for *this window only*. Write your outputs here.
* `config` — the analysis configuration for the run (window size, overlap, frequency bands, etc).
* `image_data` — scan metadata, RF data, and B-mode data (pre/post scan-conversion).

This function has no visibility into any other window — it cannot read or influence `window.results` on a sibling window. If your computation genuinely needs every window's data at once (e.g. averaging a per-window curve across the whole ROI before fitting a single aggregate value), it doesn't belong here — see "Aggregate" below.

### 2. Full segmentation (`@location('full_segmentation')`)

Runs once, on a single synthetic `Window` spanning the bounding box of every generated window (i.e. the whole ROI treated as one big window), re-sliced directly from the raw RF data — not an aggregation of the small windows' already-computed results. Same signature and semantics as "window", just called once with different bounds, via `compute_single_window()` (called after `compute_paramaps()`).

### 3. Aggregate (`@location('aggregate')`)

Runs once, after **every** small sliding-window tile has been fully processed (`compute_aggregate_vals()`, called automatically at the end of `compute_paramaps()`). Unlike the other two locations, it receives the full list of `Window` objects — real cross-window access — and writes onto a separate `aggregate_results` object (not any individual `window.results`):

```python
def METHOD_NAME(windows: List[Window], aggregate_results: BlankResults,
                    config: RfAnalysisConfig, image_data: UltrasoundRfImage, **kwargs) -> None:
```

Use this when a computation genuinely needs every window's result simultaneously — most commonly "average some per-window curve/value across the whole ROI, then fit/reduce it to one scalar for the scan" (e.g. `analysis_aggregators/ac_scan_stats.py`, which turns per-window `attenuation_coef` output into a single AC estimate; `analysis_aggregators/bsc_scan_stats.py`, which turns per-window `bsc_transmission_compensation` curves into a Lizzi-Feleppa slope/intercept/midband fit; `analysis_aggregators/homodyned_k_scan_stats.py`, which NaN-omitting-means per-window `homodyned_k_params` k/mu estimates — each mirroring a MATLAB reference driver script's own final post-processing step over all ROIs). This is *not* the common case — most QUS methods (Nakagami, H-scan, the NPS-based Lizzi-Feleppa variant, ...) are fully expressible per-window and should stay `'window'`-scoped.

**Every function declaring `@location("aggregate")` must live in `paramap/analysis_aggregators/` (or `bmode/analysis_aggregators/`)** — a sibling of `analysis_methods/`, not a subfolder nested inside it — e.g. `paramap/analysis_aggregators/ac_scan_stats.py`, not `paramap/analysis_methods/ac_scan_stats.py`. This is a hard requirement, not a style preference: it keeps "a folder under `analysis_methods/` is one plugin" unambiguous (see "Single-file vs. folder plugins" above) rather than a folder sometimes meaning a whole category of plugins, and it makes aggregate functions visually distinct in the directory tree without opening any file. It's discovered exactly the same way as `analysis_methods/` (single `.py` file or a folder with `__init__.py`), at the same nesting depth, so relative imports use the same dot-count as a flat-file plugin in `analysis_methods/` (e.g. `from ..decorators import *` and `from ....data_objs...`). An external/private aggregate plugin mirrors this too: `$QUANTUS_PLUGIN_DIR/analysis/paramap/analysis_aggregators/your_func.py`.

There are two ways to declare which window plugin(s) an aggregate function depends on:

- **Static dependency** — for an aggregate function purpose-built for one window plugin (like `ac_scan_stats`), declare it the same way any other dependency is declared: `@dependencies("attenuation_coef")`. This both auto-includes that plugin in the run if it wasn't explicitly requested, and guarantees (via the existing topological ordering) that it's fully computed on every window before your aggregate function runs.
- **Config-driven dependency** — for a *generic*, reusable aggregate function whose target window plugin isn't known until config time (e.g. `window_mean`/`window_median` in `analysis_aggregators/window_stats.py`, which can aggregate any window plugin's output), declare `@required_kwargs("source_func", "source_attr")` instead of a static `@dependencies(...)`. The pipeline validates at config-load time that `analysis_kwargs['source_func']` is actually present in `analysis_funcs` (in both `core_pipeline` and `entrypoints.analysis_step`), so a config referencing a plugin that wasn't run fails early with a clear message instead of a deep `AttributeError`. Because the output name depends on `source_attr` (e.g. `aggregate_results.nak_w_mean`), these functions declare `@output_vars()` (empty) rather than a fixed name — the `<source_attr>_<stat>` naming convention is documented in the function's own docstring instead. Note `analysis_kwargs` is one flat dict shared by every function in a run, so only one `source_func`/`source_attr` pair can be aggregated this way per run today.

**Reclaiming memory** — an aggregate function often consumes a bulky per-window intermediate (e.g. a full frequency-curve array) that nothing downstream (visualization, `descr_vals`, another aggregate function) still needs once it's been aggregated. Rather than a plugin unilaterally deciding to discard that data, it's an opt-in, per-run **config** setting: `analysis_kwargs['aggregate_evict']`, a list of `window.results` attribute names deleted from every window after all aggregate functions have run. E.g. `aggregate_evict: [att_coefs]` after running `ac_scan_stats` (the identical-across-windows `att_subf` isn't worth listing). Only mark attributes evictable that nothing else in the run still needs — in particular, never evict a scalar meant for paramap painting (`draw_paramap` reads `window.results.<param>` for every window and will break if it's gone).

## Decorators

Metadata can be added to new analysis method functions using decorators defined in [paramap/decorators.py](paramap/decorators.py).

* `dependencies` — other functions that must be run before this one (typically because this function reads their output). Works the same way across all three locations, including aggregate.
* `supported_spatial_dims` — the supported spatial dimensions of a QUS method.
* `required_kwargs` — additional variables needed by the plugin (document each in the plugin's docstring).
* `default_kwarg_vals` — default values for kwargs not provided in the analysis configuration.
* `output_vars` — the variable names written by the function (on `window.results` for window/full_segmentation functions, on `aggregate_results` for aggregate ones with a static/known output name; leave empty for a generic aggregate function with a config-dependent output name).
* `location` — where in the pipeline the function runs: `'window'`, `'full_segmentation'`, or `'aggregate'` (see above). Defaults to `['window', 'full_segmentation']` if undeclared.

## YAML configs

No schema change is needed to use an aggregate function — its name is just another entry in a config's existing `analysis_funcs` list, and any kwargs it needs (including `source_func`/`source_attr` for a generic aggregator, or `aggregate_evict`) go in the existing `analysis_kwargs` block. See `configs/siemens_rf_att.yaml` for a worked example pairing `attenuation_coef` (window) with `ac_scan_stats` (aggregate).

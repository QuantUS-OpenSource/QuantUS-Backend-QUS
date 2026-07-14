# Architecture

QuantUS-Plugins is a quantitative ultrasound (QUS) analysis framework built on an extensible plugin architecture. It loads RF ultrasound data from various manufacturers, applies configurable QUS analysis methods over a segmented region using a sliding-window technique, and exports parametric maps, visualizations, and numerical features. It's used both as a library (via the GUI in the separate QuantUS repo) and standalone via CLI/YAML config/scripting.

This document is the durable architecture reference for the codebase — how the six plugin stages fit together, and the core data objects threaded through all of them. See [AGENTS.md](../AGENTS.md) for how to actually add a new plugin.

## Pipeline stages

1. **Image loading** — load B-mode and RF signal data from a manufacturer-specific file format.
2. **Segmentation loading** — draw or load a saved binary mask/spline over the region to analyze.
3. **Analysis** — parametric-map analysis using the sliding-window technique over the segmented region.
4. **Analysis configuration** — scan metadata and sliding-window parameters (window size, overlap, frequency bands) used by analysis.
5. **Visualizations** — save parametric map outputs and additional plots from the analysis results.
6. **Data export** — extract numerical features from parametric maps and write them out (CSV-only today).

`quantus/full_workflow.py`'s `core_pipeline()` runs all six stages in order from a single config (YAML, dict, or argparse namespace); `quantus/entrypoints.py` exposes each stage as an individually callable function for custom/batch scripting (see `quantus/processing/` and `CLI-Demos/*.ipynb`).

## Plugin discovery pattern

Every pipeline stage follows the same convention, repeated across six plugin categories. Understanding one means understanding all of them:

- **`options.py`** (one per stage, e.g. `quantus/image_loading/utc_loaders/options.py`, `quantus/analysis/options.py`) — dynamically discovers plugins by scanning a folder for subfolders/files and importing them; returns a dict mapping plugin name → class/function + metadata (supported extensions, spatial dims, required kwargs, etc). This is the single source of truth for "what plugins exist" — never hardcode a plugin list, always go through `get_*` in the relevant `options.py`.
- **`transforms.py`** (per stage) — shared helper functions usable across multiple plugins in that stage.
- **`decorators.py`** (per stage, where applicable) — attaches metadata to plugin functions (e.g. `dependencies`, `supported_spatial_dims`, `required_kwargs`, `default_kwarg_vals`, `output_vars`, `location` in `quantus/analysis/paramap/decorators.py`). `options.py` reads these attributes off the function object rather than requiring a registration call.
- **`quantus/plugin_utils.py`** — shared discovery helpers (`discover_plugin_classes`, `discover_marked_functions`) and the `attr_decorator` factory underlying every stage's `options.py`/`decorators.py`. Also defines the external plugin directory: every `get_*` function additionally scans `QUANTUS_PLUGIN_DIR` (default `.quantus/plugins/` at the repo root, via `get_quantus_home_dir()`), merging what it finds in additively alongside the built-in plugins (external wins on a name collision, with a warning). This is how private/proprietary plugins are distributed — dropped in outside the repo's git tree — and the same mechanism a future GUI would use to load plugins dynamically.
- **`PluginParam`/`get_plugin_params()`** (also in `plugin_utils.py`) — normalizes a plugin's kwargs/defaults decorator attributes (whichever pair a given stage uses — `required_kwargs`/`default_kwarg_vals`, or `gui_kwargs`/`default_gui_kwarg_vals`) into one stable `List[PluginParam]` shape. Intended as the introspectable contract a future GUI would build dynamic form fields from, without hardcoding which attribute names apply to which stage.

The six plugin stages, each with its own README documenting the exact function/class signature new plugins must implement:
1. **Image loading** (`quantus/image_loading/`) — parsers as folders under `utc_loaders/` (e.g. `clarius_rf/`, `canon_iq/`), each with a `main.py` exposing `EntryClass` (subclass of `UltrasoundRfImage`).
2. **Segmentation loading** (`quantus/seg_loading/`) — single-file plugins under `seg_loaders/`, each a function returning a `BmodeSeg`.
3. **Analysis config loading** (`quantus/analysis_config/`) — single-file plugins under `utc_config/config_loaders/`, each a function returning an `RfAnalysisConfig`.
4. **Analysis** (`quantus/analysis/`) — QUS methods as single-file plugins under `paramap/analysis_methods/` (or `bmode/analysis_methods/`), each a function taking `(scan_rf_window, phantom_rf_window, window, config, image_data, **kwargs)` and writing results onto the `window` object.
5. **Visualizations** (`quantus/visualizations/`) — plugins under `paramap/visualization_funcs/`, each a function `(analysis_obj, dest_folder, **kwargs)`.
6. **Data export** (`quantus/data_export/`) — plugins under `csv/export_funcs/`, each a function `(visualizations_obj, data_dict, **kwargs)` (CLI-only, no GUI support currently).

## Core data objects (`quantus/data_objs/`)

These are the objects threaded through every pipeline stage; all plugins read from and/or populate them:
- `UltrasoundRfImage` (`image.py`) — RF + B-mode data, resolution, and scan-conversion state. All RF analysis must operate on **pre-scan-conversion** data; `sc_`-prefixed attributes hold the post-scan-conversion (polar→cartesian) counterparts for display only.
- `BmodeSeg` (`seg.py`) — segmentation mask/splines, with `sc_` variants mirroring the same pre/post scan-conversion split. `frame` selects the relevant frame for multi-frame scans.
- `RfAnalysisConfig` (`analysis_config.py`) — scan metadata and sliding-window parameters (window size, overlap, frequency bands). 3D-only fields (`cor_win_size`, `coronal_overlap`) stay `None` for 2D scans.
- `ParamapAnalysisBase`, `ParamapDrawingBase`, `BaseDataExport` (`analysis.py`, `visualizations.py`, `data_export.py`) — base classes driving the analysis/visualization/export stages respectively.

2D vs 3D is controlled by `spatial_dims` on the image loader's `EntryClass`; plugin functions that only support one dimensionality declare it via the `supported_spatial_dims` decorator, and `entrypoints.py`/`full_workflow.py` validate this before running.

Cross-stage coupling through these objects is intentional, not accidental: a new analysis plugin, for example, necessarily consumes `BmodeSeg` and `UltrasoundRfImage` types defined centrally here, since the sliding-window technique needs both the segmentation and the underlying RF data together. This is the one place plugin authors need to look outside their own stage.

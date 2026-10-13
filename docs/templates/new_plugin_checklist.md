# New plugin checklist

Fill in the row for the stage you're adding a plugin to. Full details for each stage are in its README (linked below) — this is a fast sanity check, not a replacement for reading it.

| Stage | Where it goes | Entry shape | README |
|---|---|---|---|
| Image loading | New folder: `quantus/image_loading/utc_loaders/<name>/main.py` | Class named `EntryClass`, subclass of `UltrasoundRfImage`, with `extensions`, `spatial_dims`, `gui_kwargs`, `cli_kwargs`, `default_gui_kwarg_vals`, `default_cli_kwarg_vals` class attributes | [quantus/image_loading/README.md](../../quantus/image_loading/README.md) |
| Segmentation loading | New file: `quantus/seg_loading/seg_loaders/<name>.py` | Function `<name>(image_data, seg_path, **kwargs) -> BmodeSeg`, decorated with `@extensions(...)` from `seg_loading/decorators.py` | [quantus/seg_loading/README.md](../../quantus/seg_loading/README.md) |
| Analysis config loading | New file: `quantus/analysis_config/utc_config/config_loaders/<name>.py` | Function `<name>(analysis_path, **kwargs) -> RfAnalysisConfig`, decorated with `@extensions(...)` (and optionally `@gui_kwargs(...)`, `@default_gui_kwarg_vals(...)`) | [quantus/analysis_config/README.md](../../quantus/analysis_config/README.md) |
| Analysis (QUS method) | New file: `quantus/analysis/paramap/analysis_methods/<name>.py` (or a folder `<name>/__init__.py` if bundling a sibling data asset); or `bmode/analysis_methods/`. An `aggregate`-located function instead goes in the sibling folder `analysis_aggregators/<name>.py` — required, not optional, so it's visually distinct from window/full_segmentation plugins. | Function `<name>(scan_rf_window, phantom_rf_window, window, config, image_data, **kwargs) -> None`, writes results onto `window`; decorated with `@output_vars(...)` at minimum. (An `aggregate`-located function instead takes `(windows, aggregate_results, config, image_data, **kwargs)` and lives in `analysis_aggregators/` — see the README.) | [quantus/analysis/README.md](../../quantus/analysis/README.md) |
| Visualizations | New file: `quantus/visualizations/paramap/visualization_funcs/<name>.py` | Function `<name>(analysis_obj, dest_folder, **kwargs) -> None` | [quantus/visualizations/README.md](../../quantus/visualizations/README.md) |
| Data export | New file: `quantus/data_export/csv/export_funcs/<name>.py` | Function `<name>(visualizations_obj, data_dict, **kwargs) -> None`, decorated with `@required_kwargs(...)` | [quantus/data_export/README.md](../../quantus/data_export/README.md) |

## Before you're done

- [ ] Read the stage README linked above for the exact contract (not all six are identical — e.g. seg loaders are wrapped in a dict, unlike other stages).
- [ ] Used an existing plugin in the same stage as a template.
- [ ] Confirmed the plugin is picked up: run `pytest` (the discovery smoke tests will exercise it automatically), or call the stage's `get_*` function directly and check it's in the returned dict.
- [ ] If this is a private/proprietary plugin, put it under `QUANTUS_PLUGIN_DIR` instead of this repo — see [AGENTS.md](../../AGENTS.md#private--external-plugins).

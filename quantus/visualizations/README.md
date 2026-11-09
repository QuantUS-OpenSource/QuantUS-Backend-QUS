# Visualizations

This directory contains plugins for generating visualizations from ultrasound analysis results. The visualization system allows you to extend QuantUS with custom plots, parameter maps, and visual outputs by adding files to `paramap/visualization_funcs`.

Visualization plugins transform numerical analysis results into visual representations such as parameter maps, plots, overlays, and statistical graphics. Each visualization function creates and saves specific types of visual outputs for research, clinical, or presentation purposes.

## Output location

By default, parametric map outputs (`paramaps`) are saved to `.quantus/visualization_results/` at the repo root (see `get_quantus_home_dir()` in `quantus/plugin_utils.py`), keeping generated output out of the repo's tracked files. Pass `paramap_folder_path` in `visualization_kwargs` to override this — an absolute path is used as-is, while a relative path is resolved against the repository root regardless of the current working directory.

## External / private plugins

Plugins don't have to live in this repo. Set `QUANTUS_PLUGIN_DIR` (defaults to `.quantus/plugins/` at the repo root) to a directory containing a `visualizations/` subfolder mirroring this layout (e.g. `$QUANTUS_PLUGIN_DIR/visualizations/paramap/visualization_funcs/your_plot.py`), and it's picked up alongside the built-ins — no changes to this repo needed. This is how private/proprietary visualizations are meant to be distributed (an approved user drops a received plugin file in) and how a future GUI can load plugins dynamically. A plugin here with the same name as a built-in one overrides it (with a warning); everything else stays available.

## Plugin Implementation

### Plugin Structure

Each visualization plugin should be placed in the [quantus/visualizations/paramap/visualization_funcs](paramap/visualization_funcs) folder as a new .py file containing a function. Specifically, the new function must be in the following form:

```python
def VIS_NAME(analysis_obj: Any, dest_folder: str, **kwargs):
```

where `VIS_NAME` is the name of your visualization plugin. The inputs contain the standard inputs for a visualization function, and the `kwargs` variable can be used to add any additional input variables that may be needed.

* The `dest_folder` input contains the name of the folder in which all visualizations should be exported to.

### Decorators

Metadata can be added to new segmentation parsing functions using decorators defined in [quantus/visualizations/paramap/decorators.py](paramap/decorators.py).

* The `dependencies` decorator specifies the other functions which must be run before the current function. Typically, this is because the current function depends on the outputs of another function.
* The `gui_kwargs` decorator provides keyword arguments for initialization accessible from the GUI.
* The `default_gui_kwarg_vals` decorator specifies default values for each keyword argument in `gui_kwargs`.

Note visualization kwargs aren't currently implemented in the GUI, so the `gui_kwargs` are only accessible via the CLI for now.

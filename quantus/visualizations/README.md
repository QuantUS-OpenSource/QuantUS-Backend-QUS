# Visualizations

This directory contains plugins for generating visualizations from ultrasound analysis results. The visualization system allows you to extend QuantUS with custom plots, parameter maps, and visual outputs by adding files to `paramap/visualization_funcs`.

Visualization plugins transform numerical analysis results into visual representations such as parameter maps, plots, overlays, and statistical graphics. Each visualization function creates and saves specific types of visual outputs for research, clinical, or presentation purposes.

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

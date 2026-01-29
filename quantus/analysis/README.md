# Analysis plugins

This directly contains all QUS analysis methods used for analysis. The plugin system enables users to extend QuantUS with new QUS methods by adding files to `paramap/analysis_methods` or `bmode/analysis_methods`.

All analysis is currently based around parametric maps via the sliding window technique.

## Plugin Implementation

### Plugin Structure

Each curve definition plugin should be placed in the [quantus/analysis/paramap/analysis_methods](paramap/analysis_methods) folder as a new .py file containing a function. Specifically, the new function must be in the following form:

```python
def METHOD_NAME(scan_rf_window: np.ndarray, phantom_rf_window: np.ndarray, 
                    window: Window, config: RfAnalysisConfig, 
                    image_data: UltrasoundRfImage, **kwargs) -> None:
```

where `METHOD_NAME` is the name of your parser. The inputs contain the standard inputs for a QUS methods function, and the `kwargs` variable can be used to add any additional input variables that may be needed.

* The `scan_rf_window` input contains a single window within the segmentation derived from the sliding window technique. It can be 2D (samples, channels) or 3D (samples, channels, slices). The window is of the pre-scan conversion image, if applicable.
* The `phantom_rf_window` is the same window as `scan_rf_window` spatially, but for the inputted phantom scan.
* The `window` object is a class containing spatial metadata and previously computed results on the current window. Outputs should be added to this object.
* The `config` object contains the analysis configuration data for the current analysis run.
* The `image_data` object contains metadata, RF data, and B-Mode data (pre- and post- scan conversion if applicable) to be used for analysis.

### Decorators

Metadata can be added to new segmentation parsing functions using decorators defined in [src/time_series_analysis/curve_types/decorators.py](decorators.py).

* The `dependencies` decorator specifies the other functions which must be run before the current function. Typically, this is because the current function depends on the outputs of another function.
* The `supported_spatial_dims` decorator specifies the supported spatial dimensions of a QUS method.
* The `required_kwargs` decorator lists the additional variables needed for the QUS plugin. Detailed explanations of each kwarg should be listed in the docstring of the plugin function.
* The `default_kwarg_vals` decorator specifies default values for kwargs if they are not provided in the analysis configuration.
* The `output_vars` decorator specifies the variable names written to in the function.
* The `location` decorator specifies the locations a function can be called from in the parametric map analysis pipeline. Possible locations include: 'window', 'full_segmentation'.

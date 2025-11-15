# Data Export Plugins

This directory contains plugins for exporting numerical analysis results to various file formats. The data export system allows you to extend QuantUS with new export formats and custom data processing by adding functions to `csv/functions.py`. CSV export is currently the only supported format, but more can be added.

Data export plugins take analysis results and format them for external use. Each export function processes the computed analysis parameters and saves them in a structured format for further analysis, reporting, or integration with other tools.

## Plugin Architecture

### Plugin Structure

Each data export plugin should be placed in the [quantus/data_export/csv/functions.py](csv/functions.py) file as a new function. Specifically, the new function must be in the following form:

```python
def EXPORT_FUNC(visualizations_obj: ParamapDrawingBase, data_dict: Dict[str, str], **kwargs) -> None:
```

where `EXPORT_FUNC` is the name of your export function. The inputs contain the standard inputs for a data export function, and the `kwargs` variable can be used to add any additional input variables that may be needed.

* The `data_dict` input contains the data to be exported, structured as a dictionary. This should be added to in the plugin function. Exporting will be handled by the core export framework.

### Decorators

Metadata can be added to new segmentation parsing functions using decorators defined in [quantus/data_export/csv/decorators.py](csv/decorators.py).

* The `required_kwargs` decorator specifies any additional keyword arguments that must be provided for the function to operate correctly.

## Notes

This functionality is currently limited to CLI usage. GUI support may be added in the future.

# Analysis configuration loading

Analysis configuration loading plugins load scan metadata and parametric map parameters before analysis. Plugins support data loading from different file formats.

New plugins can be added to the [quantus/analysis_config/utc_config/functions.py](utc_config/functions.py) file as a new function, and will extend the capabilities of QuantUS without any additional programming required.

## Plugin Implementation

### Core data class

All analysis configuration parsers populate the `RfAnalysisConfig` class as defined in [quantus/data_objs/analysis_config.py](../data_objs/analysis_config.py).

```python
class RfAnalysisConfig:
    """
    Class to store configuration data for RF analysis.
    """

    def __init__(self):
        self.transducer_freq_band: List[int]  # [min, max] (Hz)
        self.analysis_freq_band: List[int]  # [lower, upper] (Hz)
        self.sampling_frequency: int  # Hz
        self.ax_win_size: float  # axial length per window (mm)
        self.lat_win_size: float  # lateral width per window (mm)
        self.window_thresh: float  # % of window area required to be in ROI
        self.axial_overlap: float  # % of ax window length to move before next window
        self.lateral_overlap: float  # % of lat window length to move before next window
        self.center_frequency: float  # Hz
        
        # 3D scan parameters
        self.cor_win_size: float  # coronal length per window (mm)
        self.coronal_overlap: float # % of cor window length to move before next window

```

Note 3D scan parameters are only required for 3D scans, and can be left as `None` for 2D scans.

### Plugin Structure

Each analysis configuration loading plugin should be placed in the [quantus/analysis_config/utc_config/functions.py](functions.py) file as a new function. Specifically, the new function must be in the following form:

```python
def CONFIG_LOADER_NAME(analysis_path: str, **kwargs) -> RfAnalysisConfig:
```

where `CONFIG_LOADER_NAME` is the name of your parser. The inputs contain the standard parser inputs for an analysis configuration parser, and the `kwargs` variable can be used to add any additional input variables that may be needed.

### Decorators

Metadata can be added to new segmentation parsing functions using decorators defined in [quantus/seg_loading/decorators.py](decorators.py).

* The `extensions` decorator, which specifies the required suffix of potential segmentation files.
* The `gui_kwargs` decorator provides keyword arguments for initialization accessible from the GUI.
* The `default_gui_kwarg_vals` decorator specifies default values for each keyword argument in `gui_kwargs`.

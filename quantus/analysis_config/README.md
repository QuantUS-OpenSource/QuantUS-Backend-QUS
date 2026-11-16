# Analysis configuration loading

Analysis configuration loading plugins load scan metadata and parametric map parameters before analysis. Plugins support data loading from different file formats.

New plugins can be added to the [quantus/analysis_config/utc_config/config_loaders](utc_config/config_loaders) folder as a new .py file containing a function, and will extend the capabilities of QuantUS without any additional programming required.

## External / private plugins

Plugins don't have to live in this repo. Set `QUANTUS_PLUGIN_DIR` (defaults to `.quantus/plugins/` at the repo root) to a directory containing an `analysis_config/` subfolder with plugin files in the same form described below (e.g. `$QUANTUS_PLUGIN_DIR/analysis_config/your_loader.py`), and it's picked up alongside the built-ins — no changes to this repo needed. This is how private/proprietary loaders are meant to be distributed (an approved user drops a received plugin file in) and how a future GUI can load plugins dynamically. A plugin here with the same name as a built-in one overrides it (with a warning); everything else stays available.

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
        self.ax_win_size_units: str  # WIN_SIZE_UNITS_MM or WIN_SIZE_UNITS_WAVELENGTH
        self.lat_win_size_units: str  # WIN_SIZE_UNITS_MM, WIN_SIZE_UNITS_WAVELENGTH, or WIN_SIZE_UNITS_A_LINES
        self.ax_win_size: float  # axial length per window (in ax_win_size_units)
        self.lat_win_size: float  # lateral width per window (in lat_win_size_units)
        self.speed_of_sound: float  # speed of sound through tissue (m/s), used for wavelength units
        self.window_thresh: float  # % of window area required to be in ROI
        self.axial_overlap: float  # % of ax window length to move before next window
        self.lateral_overlap: float  # % of lat window length to move before next window
        self.center_frequency: float  # Hz
        
        # 3D scan parameters
        self.cor_win_size: float  # coronal length per window (mm)
        self.coronal_overlap: float # % of cor window length to move before next window

```

Note 3D scan parameters are only required for 3D scans, and can be left as `None` for 2D scans.

Window size units are named constants defined in [quantus/data_objs/analysis_config.py](../data_objs/analysis_config.py): `WIN_SIZE_UNITS_MM` (`"mm"`), `WIN_SIZE_UNITS_WAVELENGTH` (`"number of wavelength"`, converted to mm via `speed_of_sound / center_frequency`), and `WIN_SIZE_UNITS_A_LINES` (`"A-Lines"`, lateral only). They have no defaults, so every 2D config loader must set `ax_win_size_units`, `lat_win_size_units`, and `speed_of_sound` (see `clarius_C3_config.py` for a wavelength/A-line example). Units currently apply to 2D windowing only.

### Plugin Structure

Each analysis configuration loading plugin should be placed in the [quantus/analysis_config/utc_config/config_loaders](utc_config/config_loaders) file as a new .py file containing a function. Specifically, the new function must be in the following form:

```python
def CONFIG_LOADER_NAME(analysis_path: str, **kwargs) -> RfAnalysisConfig:
```

where `CONFIG_LOADER_NAME` is the name of your parser. The inputs contain the standard parser inputs for an analysis configuration parser, and the `kwargs` variable can be used to add any additional input variables that may be needed.

### Decorators

Metadata can be added to new analysis configuration loader functions using decorators defined in [quantus/analysis_config/utc_config/decorators.py](utc_config/decorators.py).

* The `extensions` decorator, which specifies the required suffix of potential segmentation files.
* The `gui_kwargs` decorator provides keyword arguments for initialization accessible from the GUI.
* The `default_gui_kwarg_vals` decorator specifies default values for each keyword argument in `gui_kwargs`.

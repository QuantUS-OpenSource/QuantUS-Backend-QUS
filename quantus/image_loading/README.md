# Image loading

Image loading plugins include all data loading parsers for QuantUS. These parsers can read RF scans from different manufacturers and file formats. B-Mode and RF data for analysis are stored separately in the these parsers for downstream flexibility.

New plugins can be dropped into this folder to extend the capabilities of QuantUS without any additional programming required.

## Plugin implementation

### Core data class

All QUS parsers load data into the `UltrasoundRfImage` class as defined in [quantus/data_objs/image.py](../data_objs/image.py). Thus, the class entrypoint of each parser is a child class of the `UltrasoundRfImage`.

```python
class UltrasoundRfImage:
    """
    Class for ultrasound RF image data.
    """

    def __init__(self, scan_path: str, phantom_path: str):
        # RF data
        self.rf_data: np.ndarray
        self.phantom_rf_data: np.ndarray
        self.bmode: np.ndarray
        self.axial_res: float # mm/pix
        self.lateral_res: float # mm/pix
        self.scan_name = Path(scan_path).stem
        self.phantom_name = Path(phantom_path).stem
        self.scan_path = scan_path
        self.phantom_path = phantom_path
        self.spatial_dims: int
        
        # Scan conversion parameters
        self.sc_bmode: np.ndarray = None
        self.xmap: np.ndarray # sc (y,x) --> preSC x
        self.ymap: np.ndarray # sc (y,x) --> preSC y
        self.width: float # deg
        self.tilt: float
        self.start_depth: float # mm
        self.end_depth: float # mm
        self.sc_axial_res: float # mm
        self.sc_lateral_res: float
        
        # 3D scan parameters
        self.coronal_res: float = None # mm/pix
        self.depth: float # depth in mm
        
        # 3D scan conversion parameters
        self.sc_bmode: np.ndarray = None
        self.coord_map_3d: np.ndarray # maps (z,y,x) in SC coords to (x,y) preSC coord
        self.sc_axial_res: float # mm/pix
        self.sc_lateral_res: float # mm/pix
        self.sc_coronal_res: float # mm/pix
        self.sc_params_3d = None
```

Parsers handling two spatial dimensions do not need to fill out the 3D attributes. The `sc` prefix in each attribute refers to scan conversion, the optional transformation from polar to cartesian coordinates for images taken with non-linear probes.

Managing the state before scan conversion is crucial as all RF analysis must be done in the non-scan converted coordinates.

## Plugin structure

Each image loading plugin should be placed in `quantus/image_loading/your_plugin_name/` with the following structure:

```
your_plugin_name/
├── main.py          # Required: EntryClass implementation
├── parser.py        # Recommended: Core parsing logic
├── objects.py       # Optional: Custom data structures
└── utils.py         # Optional: Helper functions
```

The child class of the `UltrasoundRfImage` base class should be named `EntryClass`, and this is the final class which will interact with the rest of the analysis workflow.

### Additional attributes

In addition to the default methods of the `UltrasoundImage` base class, the plugin `EntryClass` must also contain three additional class attributes to finish interfacing with the rest of the workflow.

| Attribute | Type | Description |
|-----------|------|-------------|
| `extensions` | List[str] | Supported file extensions (e.g., `[".bin", ".dat"]`) |
| `spatial_dims` | int | 2 for 2D data, 3 for 3D data |
| `gui_kwargs` | List[str] | Keyword arguments for initialization accessible from the GUI |
| `cli_kwargs` | List[str] | Keyword arguments for initialization not accessible from the GUI |
| `default_gui_kwarg_vals` | List[Any] | Default values for each keyword argument in `gui_kwargs` |
| `default_cli_kwarg_vals` | List[Any] | Default values for each keyword argument in `cli_kwargs` |

See the existing [clarius_rf](utc_loaders/clarius_rf/) plugin for an example. Once implemented, this parser will be available from the GUI and CLI to run custom analysis with.

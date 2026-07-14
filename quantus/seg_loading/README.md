# Segmentation Loading

Segmentation loading plugins load binary masks to specify where in the loaded scan to analyze. Plugins support data loading from different file formats.

New plugins can be added to the [quantus/seg_loading/seg_loaders](seg_loaders) folder as a new .py file containing a function, and will extend the capabilities of QuantUS without any additional programming required.

## External / private plugins

Plugins don't have to live in this repo. Set `QUANTUS_PLUGIN_DIR` (defaults to `.quantus/plugins/` at the repo root) to a directory containing a `seg_loading/` subfolder with plugin files in the same form described below (e.g. `$QUANTUS_PLUGIN_DIR/seg_loading/your_loader.py`), and it's picked up alongside the built-ins — no changes to this repo needed. This is how private/proprietary loaders are meant to be distributed (an approved user drops a received plugin file in) and how a future GUI can load plugins dynamically. A plugin here with the same name as a built-in one overrides it (with a warning); everything else stays available.

## Plugin Implementation

### Core data class

All segmentation parsers populate the `BmodeSeg` class as defined in [quantus/data_objs/seg.py](../data_objs/seg.py).

```python
class BmodeSeg:
    """
    Class for ultrasound RF image data.
    """

    def __init__(self):
        self.seg_name: str
        self.splines: List[np.ndarray] # [X, Y, (Z)]
        self.seg_mask: np.ndarray
        self.sc_seg_mask: np.ndarray
        self.frame: int
        
        # Scan conversion parameters
        self.sc_splines: List[np.ndarray] # [X, Y, (Z)]
```

For 3D segmentations, the Z dimension is included in the splines list. For scans without scan conversion, `sc_splines` and `sc_seg_mask` can be left as `None`. For scans with multiple frames, `frame` should be set to the appropriate frame index. Otherwise, it can be left as `None`.

### Plugin Structure

Each segmentation loading plugin should be placed in the [quantus/seg_loading/seg_loaders](seg_loaders) folder as a new .py file containing a function. Specifically, the new function must be in the following form:

```python
def SEG_LOADER_NAME(image_data: UltrasoundImage, seg_path: str, **kwargs) -> BmodeSeg:
```

where `SEG_LOADER_NAME` is the name of your parser. The inputs contain the standard parser inputs for a segmentation parser, and the `kwargs` variable can be used to add any additional input variables that may be needed.

### Decorators

Metadata can be added to new segmentation parsing functions using decorators defined in [quantus/seg_loading/decorators.py](decorators.py).

Currently, the only implemented decorator for this parser is the `extensions` decorator, which specifies the required suffix of potential segmentation files.

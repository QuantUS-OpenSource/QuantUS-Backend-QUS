# QuantUS-Plugins

A quantitative ultrasound (QUS) analysis framework built on an extensible plugin architecture. It processes ultrasound RF data from various manufacturers and extracts tissue characterization parameters while being accessible through both a graphical user interface (GUI) and command-line interface (CLI).

## Workflow overview

QuantUS QUS analysis follows a sequential and customizable analysis workflow, supporting both 2D and 3D frame dimensions.

1. `Image loading` - Load brightness mode (B-Mode) and RF data from custom data source and manufacturer.
2. `Segmentation loading` - Draw or load saved segmentation to use for analysis. Segmentations are saved as binary masks.
3. `Analysis type` - Parametric map analysis using the sliding window technique on the segmentation mask is the only currently supported analysis type.
4. `QUS Methods Selection` - Customize the QUS methods to run on inputted image and segmentation. New QUS methods can easily be injected into the pipeline for experimentation.
5. `Analysis Configuration` - Customize the scan metadata and parametric map parameters before analysis.
6. `Visualizations` - Save parametric map outputs and visualizations at the end of your analysis. Customize and additional visualizations to output.
7. `Numerical data exporting` - Customize numerical features to extract from generated parametric maps and how to export.

## Installation

### Requirements

- Python 3.10

### Steps

To install the QuantUS platform, follow these steps. Let `PYTHON310` be the path to your Python3.10 interpreter.

1. **Clone the repository**

```bash
git clone https://github.com/TUL-Dev/QuantUS-Plugins
cd QuantUS-Plugins
```

2. **Install the package**

```bash
$PYTHON310 -m pip install virtualenv
$PYTOHN310 -m virtualenv .venv
source .venv/bin/activate                           # Unix
.venv\Scripts\activate                              # Windows (cmd)
pip install --upgrade pip setuptools wheel
pip install numpy
pip install "napari[all]"
pip install -r requirements.txt
pip install -e .
./saveQt.sh                                         # Unix
.\saveQt.sh                                         # Windows (cmd)
```

## Usage

### Interfaces

1. **Command-line interface (CLI)**

The CLI enables users to run the entire workflow at once and save results as needed. Sample workflow configurations are in the `configs` folder. This interface is ideal for playing with different settings and rapidly executing different analysis runs on a single RF scan.

This entrypoint can be accessed using

```bash
# Using .venv virtual environment
quantus $CONFIGPATH
```

2. **Scripting**

Python entrypoints to the complete workflow are ideal for supporting batch processing applications. As shown in the examples in `quantus/processing/`, you can write scripts which iterate through your entire dataset and analyze each scan/segmentation pair.

3. **Graphical user interface (GUI)**

This GUI currently supports the complete analysis pipeline except for numerical data exporting. The GUI is accessible using

```bash
# Using .venv virtual environment
python quantus/gui/main.py
```

4. **Parametric map viewing (3D)**

3D parametric maps generated from this workflow are exported as `.npy` arrays, as they can't be visualized within a JPEG image. A tutorial for visualizing these volumes is at the end of the `CLI-Demos/utc_demo_3d.ipynb` notebook.

5. **Python module (Advanced)**

Examples in `CLI-Demos` illustrate how each individual step in the workflow can be accessed via a packaged Python entrypoint. This can be used for advanced workflow customization directly in Python.

### Recommended workflow

For projects starting with just RF data and no segmentations, the recommended usage would be to first draw and save segmentations for each scan using the GUI. Next an ideal analysis config can be found and saved interactively using the GUI. The CLI can also support this config optimization if numerical features of parametric maps are needed. Last, batch processing with the finalized config can be run with scripting.

## Architecture overview

QuantUS follows a modular plugin-based architecture with clear separation of concerns to support this workflow. The remainder of the documentation in this repository is designed for develoeprs who want to understand the codebase structure and contribute new functionality.

```
QuantUS-Plugins/
├── quantus/                    # Main package
│   ├── gui/                   # GUI application (PyQt6)
│   ├── image_loading/         # Image loading plugins
│   ├── seg_loading/          # Segmentation loading plugins
│   ├── analysis/             # Analysis algorithms
│   ├── visualizations/       # Visualization tools
│   ├── data_export/          # Data export plugins
│   ├── data_objs/            # Core data structures
|   ├── processing/             # Example processing pipelines for batch processing
|   ├── entrypoints.py          # Entrypoints for individual workflow steps
|   └── full_workflow.py        # CLI interfaces and entrypoints for entire workflow 
├── configs/                  # Example configuration files
└──  CLI-Demos/               # Jupyter notebook examples
```

### Additional documentation

More information about each of these sections can be found in the README file of each folder. Note in each folder, the `options.py` file contains all functions necessary to gather all currently loaded plugins available for the workflow to use, and the `transforms.py` file contains all functions which may be useful across multiple plugins.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

For a list of improvements or bugs to work on, see the [contribution page](https://docs.google.com/spreadsheets/d/1SuCyl4_AbHcLCAIXT1xjhPTJbUcbLshlxO7MKO1gPMU/edit?usp=sharing).

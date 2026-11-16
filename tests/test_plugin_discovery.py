"""Smoke tests for the six plugin-discovery stages.

These don't run the QUS pipeline (no sample data ships with this repo) — they
verify that every stage's options.py can import cleanly, finds at least one
plugin, and that each discovered plugin carries the attributes the pipeline
(quantus/full_workflow.py's core_pipeline) relies on. This is the cheapest
possible net against a plugin silently failing discovery, which is the most
common real-world break in this architecture (see CLAUDE.md's plugin
discovery pattern section).
"""

from quantus.data_objs.image import UltrasoundRfImage
from quantus.data_objs.analysis import ParamapAnalysisBase
from quantus.data_objs.visualizations import ParamapDrawingBase
from quantus.data_objs.data_export import BaseDataExport

from quantus.image_loading.utc_loaders.options import get_scan_loaders
from quantus.seg_loading.options import get_seg_loaders
from quantus.analysis_config.utc_config.options import get_config_loaders
from quantus.analysis.options import get_analysis_types
from quantus.visualizations.options import get_visualization_types
from quantus.data_export.options import get_data_export_types


def test_scan_loaders_discovered():
    loaders = get_scan_loaders()
    assert loaders, "No scan loaders discovered"
    for name, entry in loaders.items():
        assert issubclass(entry['cls'], UltrasoundRfImage), f"{name}'s cls is not an UltrasoundRfImage subclass"
        assert entry['file_exts'], f"{name} has no file_exts"
        assert entry['spatial_dims'] in (2, 3), f"{name} has invalid spatial_dims: {entry['spatial_dims']}"


def test_seg_loaders_discovered():
    loaders = get_seg_loaders()
    assert loaders, "No segmentation loaders discovered"
    for name, entry in loaders.items():
        assert callable(entry['func']), f"{name}'s func is not callable"
        assert entry['exts'], f"{name} has no exts"


def test_config_loaders_discovered():
    loaders = get_config_loaders()
    assert loaders, "No analysis config loaders discovered"
    for name, loader in loaders.items():
        assert callable(loader), f"{name} is not callable"
        # An empty tuple is valid here (means "no extension restriction" to core_pipeline);
        # only a missing attribute (the @extensions decorator never applied) is an error.
        assert hasattr(loader, 'supported_extensions'), f"{name} has no supported_extensions"


def test_analysis_types_discovered():
    types, functions = get_analysis_types()
    assert types, "No analysis types discovered"
    for type_name, entry_class in types.items():
        assert issubclass(entry_class, ParamapAnalysisBase), f"{type_name} is not a ParamapAnalysisBase subclass"
        assert functions.get(type_name), f"Analysis type '{type_name}' has no analysis methods"
        for func_name, func in functions[type_name].items():
            assert callable(func), f"{func_name} is not callable"
            assert hasattr(func, 'outputs'), f"{func_name} has no output_vars"


def test_visualization_types_discovered():
    types, functions = get_visualization_types()
    assert types, "No visualization types discovered"
    for type_name, entry_class in types.items():
        assert issubclass(entry_class, ParamapDrawingBase), f"{type_name} is not a ParamapDrawingBase subclass"
        assert functions.get(type_name), f"Visualization type '{type_name}' has no visualization funcs"


def test_data_export_types_discovered():
    types, functions = get_data_export_types()
    assert types, "No data export types discovered"
    for type_name, entry_class in types.items():
        assert issubclass(entry_class, BaseDataExport), f"{type_name} is not a BaseDataExport subclass"
        assert functions.get(type_name), f"Data export type '{type_name}' has no export funcs"
        for func_name, func in functions[type_name].items():
            assert callable(func), f"{func_name} is not callable"
            assert hasattr(func, 'required_kwargs'), f"{func_name} has no required_kwargs"

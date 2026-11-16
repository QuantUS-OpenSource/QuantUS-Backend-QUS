from pathlib import Path

from argparse import ArgumentParser

import importlib
from . import seg_loaders
from ..plugin_utils import get_external_plugin_root, import_external_module, merge_external_plugins

def seg_loader_args(parser: ArgumentParser):
    parser.add_argument('seg_path', type=str, help='Path to segmentation file')
    parser.add_argument('--seg_type', type=str, default='pkl_roi',
                        help='Segmentation loader to use. Available options: ' + ', '.join(get_seg_loaders().keys()))
    parser.add_argument('--seg_loader_kwargs', type=str, default='{}',
                        help='Segmentation kwargs in JSON format needed for analysis class.')

def _collect_seg_loaders(directory: Path, import_module_fn) -> dict:
    functions = {}
    for file in sorted(directory.iterdir()):
        if not file.is_file() or file.suffix != ".py" or file.name.startswith("_"):
            continue
        module = import_module_fn(file)
        for name, obj in module.__dict__.items():
            if type(obj) is dict:
                try:
                    if callable(obj['func']):
                        functions[name] = {'func': obj['func'], 'exts': obj['exts']}
                except KeyError:
                    pass
    return functions

def get_seg_loaders() -> dict:
    """Get scan loaders for the CLI.

    Returns:
        dict: Dictionary of scan loaders.
    """
    loaders_path = Path(seg_loaders.__file__).parent
    functions = _collect_seg_loaders(
        loaders_path, lambda file: importlib.import_module(f"{seg_loaders.__name__}.{file.stem}")
    )

    external_root = get_external_plugin_root()
    external_dir = external_root / "seg_loading" if external_root else None
    if external_dir and external_dir.is_dir():
        external = _collect_seg_loaders(external_dir, lambda file: import_external_module(external_dir, file.stem))
        merge_external_plugins(functions, external, external_dir)

    return functions

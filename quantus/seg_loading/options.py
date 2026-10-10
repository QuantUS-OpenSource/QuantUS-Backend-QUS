from pathlib import Path

from argparse import ArgumentParser

import importlib
from . import seg_loaders 

from quantus.plugin_loader import load_user_plugins

import sys as _sys

def seg_loader_args(parser: ArgumentParser):
    parser.add_argument('seg_path', type=str, help='Path to segmentation file')
    parser.add_argument('--seg_type', type=str, default='pkl_roi',
                        help='Segmentation loader to use. Available options: ' + ', '.join(get_seg_loaders().keys()))
    parser.add_argument('--seg_loader_kwargs', type=str, default='{}',
                        help='Segmentation kwargs in JSON format needed for analysis class.')
    
def get_seg_loaders() -> dict:
    """Get scan loaders for the CLI.
    
    Returns:
        dict: Dictionary of scan loaders.
    """
    functions = {}
    loaders_path = (Path(_sys._MEIPASS) / 'quantus' / 'seg_loading' / 'seg_loaders') if getattr(_sys, 'frozen', False) else Path(seg_loaders.__file__).parent
    for file in loaders_path.iterdir():
        module = importlib.import_module(f"{seg_loaders.__name__}.{file.stem}")
        for name, obj in module.__dict__.items():
            if type(obj) is dict:
                try:
                    if callable(obj['func']):
                        functions[name] = {}
                        functions[name]['func'] = obj['func']
                        functions[name]['exts'] = obj['exts'] 
                except KeyError:
                    pass
            
    for _module in load_user_plugins("seg_loading"):
        if hasattr(_module, "SEG_LOADERS"):
            for _name, _loader in _module.SEG_LOADERS.items():
                functions[_name] = _loader

    return functions

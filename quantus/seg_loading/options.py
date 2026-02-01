from pathlib import Path

from argparse import ArgumentParser

import importlib
from . import seg_loaders 

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
    loaders_path = Path(seg_loaders.__file__).parent
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
            
    return functions

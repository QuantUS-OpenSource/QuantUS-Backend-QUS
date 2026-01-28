import importlib
import sys
from pathlib import Path

from argparse import ArgumentParser

def scan_loader_args(parser: ArgumentParser):
    parser.add_argument('scan_path', type=str, help='Path to scan signals')
    parser.add_argument('phantom_path', type=str, help='Path to phantom signals')
    parser.add_argument('scan_type', type=str,
                        help='Scan loader to use. Available options: ' + ', '.join(get_scan_loaders().keys()))
    parser.add_argument('--parser_output_path', type=str, default='parsed_data.pkl', help='Path to output parser results')
    parser.add_argument('--save_parsed_results', type=bool, default=False, 
                        help='Save parsed results to PARSER_OUTPUT_PATH')
    parser.add_argument('--scan_loader_kwargs', type=dict, default=None,
                        help='Additional arguments for the scan loader')
    
def get_scan_loaders() -> dict:
    """Get scan loaders for the CLI.
    
    Returns:
        dict: Dictionary of scan loaders.
    """
    classes = {}
    
    # 1. Load from internal-TUL if available
    project_root = Path(__file__).parents[4]
    internal_tul_path = project_root / "Internal-TUL" / "QuantUS-QUS" / "image_loading"
    
    if internal_tul_path.exists():
        if str(internal_tul_path) not in sys.path:
            sys.path.append(str(internal_tul_path))
            
        for folder in internal_tul_path.iterdir():
            if folder.is_dir() and not folder.name.startswith("_"):
                try:
                    module = importlib.import_module(f"{folder.name}.main")
                    entry_class = getattr(module, "EntryClass", None)
                    if entry_class:
                        classes[folder.name] = {}
                        classes[folder.name]['cls'] = entry_class
                        classes[folder.name]['file_exts'] = entry_class.extensions
                        classes[folder.name]['spatial_dims'] = entry_class.spatial_dims
                        classes[folder.name]['gui_kwargs'] = entry_class.gui_kwargs
                        classes[folder.name]['cli_kwargs'] = entry_class.cli_kwargs
                        classes[folder.name]['default_gui_kwarg_vals'] = entry_class.default_gui_kwarg_vals
                        classes[folder.name]['default_cli_kwarg_vals'] = entry_class.default_cli_kwarg_vals
                except Exception as e:
                    print(f"Internal module {folder.name} could not be loaded: {e}")

    # 2. Load from current directory (public loaders)
    current_dir = Path(__file__).parent
    for folder in current_dir.iterdir():
        # Check if the item is a directory and not a hidden directory
        if folder.is_dir() and not folder.name.startswith("_"):
            try:
                # Attempt to import the module
                module = importlib.import_module(f".{folder.name}.main", package=__package__)
                entry_class = getattr(module, "EntryClass", None)
                if entry_class:
                    classes[folder.name] = {}
                    classes[folder.name]['cls'] = entry_class
                    classes[folder.name]['file_exts'] = entry_class.extensions
                    classes[folder.name]['spatial_dims'] = entry_class.spatial_dims
                    classes[folder.name]['gui_kwargs'] = entry_class.gui_kwargs
                    classes[folder.name]['cli_kwargs'] = entry_class.cli_kwargs
                    classes[folder.name]['default_gui_kwarg_vals'] = entry_class.default_gui_kwarg_vals
                    classes[folder.name]['default_cli_kwarg_vals'] = entry_class.default_cli_kwarg_vals
            except ModuleNotFoundError as e:
                print(f"Module {folder.name} could not be found: {e}")
                # Handle the case where the module cannot be found
                pass
    
    return classes

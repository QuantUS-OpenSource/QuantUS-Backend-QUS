import importlib
from pathlib import Path

from argparse import ArgumentParser

from .functions import *

def config_loader_args(parser: ArgumentParser):
    parser.add_argument('config_path', type=str, help='Path to analysis config')
    parser.add_argument('--config_type', type=str, default='pkl_utc',
                        help='Analysis config loader to use. See "quantus_parse.analysis_config_loaders" in pyproject.toml for available analysis config loaders.')
    parser.add_argument('--config_kwargs', type=str, default='{}',
                        help='Analysis config kwargs in JSON format needed for analysis class.')
    
    
def get_config_loaders() -> dict:
    """Get scan loaders for the CLI.
    
    Returns:
        dict: Dictionary of scan loaders.
    """
    import sys
    import importlib
    from pathlib import Path

    functions = {}

    # 1. Load from internal-TUL if available
    project_root = Path(__file__).parents[4]
    internal_tul_path = project_root / "Internal-TUL" / "QuantUS-QUS" / "configs"

    if internal_tul_path.exists():
        if str(internal_tul_path) not in sys.path:
            sys.path.append(str(internal_tul_path))
            
        for item in internal_tul_path.iterdir():
            if item.is_file() and not item.name.startswith("_") and item.suffix == ".py":
                try:
                    module_name = item.stem
                    module = importlib.import_module(module_name)
                    for name, obj in vars(module).items():
                        if callable(obj) and hasattr(obj, 'gui_kwargs'):
                            functions[name] = obj
                except Exception as e:
                    print(f"Internal module {item.name} could not be loaded: {e}")

    # 2. Load from public functions
    for name, obj in globals().items():
        try:
            if callable(obj) and obj.__module__ == __package__ + '.functions':
                functions[name] = obj
        except KeyError:
            pass
            
    return functions

def get_required_kwargs(loader: callable) -> list:
    """Get required kwargs for a given config loader function.
    
    Args:
        loader (callable): Config loader function.  
    Returns:
        list: List of required kwargs.
    """
    gui_kwargs = loader.gui_kwargs if hasattr(loader, 'gui_kwargs') else []
    cli_kwargs = loader.cli_kwargs if hasattr(loader, 'cli_kwargs') else []
    return list(set(list(gui_kwargs) + list(cli_kwargs)))

import importlib
from pathlib import Path

from argparse import ArgumentParser

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
    functions = {}
    loaders_path = Path(__file__).parent / "config_loaders"
    for file in loaders_path.iterdir():
        module = importlib.import_module(f".config_loaders.{file.stem}", package=__package__)
        for name, obj in module.__dict__.items():
            try:
                if callable(obj) and hasattr(obj, "supported_extensions"):
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

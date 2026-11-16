from pathlib import Path

from argparse import ArgumentParser

from ...plugin_utils import discover_marked_functions, get_external_plugin_root

def config_loader_args(parser: ArgumentParser):
    parser.add_argument('config_path', type=str, help='Path to analysis config')
    parser.add_argument('--config_type', type=str, default='pkl_utc',
                        help='Analysis config loader to use. See get_config_loaders() in quantus/analysis_config/utc_config/options.py for available analysis config loaders.')
    parser.add_argument('--config_kwargs', type=str, default='{}',
                        help='Analysis config kwargs in JSON format needed for analysis class.')
    
    
def get_config_loaders() -> dict:
    """Get scan loaders for the CLI.
    
    Returns:
        dict: Dictionary of scan loaders.
    """
    loaders_path = Path(__file__).parent / "config_loaders"
    external_root = get_external_plugin_root()
    external_loaders_path = external_root / "analysis_config" if external_root else None
    return discover_marked_functions(
        __package__, ".config_loaders", loaders_path, "supported_extensions",
        external_funcs_dir=external_loaders_path,
    )

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

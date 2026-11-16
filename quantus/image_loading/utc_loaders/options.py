from pathlib import Path

from argparse import ArgumentParser

from ...plugin_utils import discover_plugin_classes

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
    current_dir = Path(__file__).parent
    entry_classes = discover_plugin_classes(
        current_dir, __package__, "main", lambda name: "EntryClass", external_stage="image_loading"
    )

    classes = {}
    for folder_name, entry_class in entry_classes.items():
        classes[folder_name] = {
            'cls': entry_class,
            'file_exts': entry_class.extensions,
            'spatial_dims': entry_class.spatial_dims,
            'gui_kwargs': entry_class.gui_kwargs,
            'cli_kwargs': entry_class.cli_kwargs,
            'default_gui_kwarg_vals': entry_class.default_gui_kwarg_vals,
            'default_cli_kwarg_vals': entry_class.default_cli_kwarg_vals,
        }

    return classes

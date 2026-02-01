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
    # Path is calculated relative to this file: 
    # engines/qus/quantus/image_loading/utc_loaders/options.py (parents[4] is engines/)
    # But Internal-TUL is actually at the git root.
    potential_project_roots = [
        Path(__file__).parents[4], # engines/qus/quantus/image_loading/utc_loaders/options.py -> engines/
        Path(__file__).parents[5], # if in engines/qus/quantus/... -> root
    ]
    
    project_root = None
    for root in potential_project_roots:
        if (root / "Internal-TUL").exists():
            project_root = root
            break
    
    if project_root is None:
        project_root = potential_project_roots[0]

    internal_tul_path = project_root / "Internal-TUL" / "QuantUS-QUS" / "image_loading"
    
    if internal_tul_path.exists():
        # Internal modules in QUS depend on quantus.full_workflow from engines/qus/quantus
        qus_engine_root = project_root / "engines" / "qus"
        if qus_engine_root.exists() and str(qus_engine_root) not in sys.path:
            sys.path.append(str(qus_engine_root))
            
        internal_tul_parent = project_root
        if internal_tul_parent.exists() and str(internal_tul_parent) not in sys.path:
            sys.path.append(str(internal_tul_parent))

        # Inject dummy sys.modules entries to satisfy relative imports in Internal-TUL
        try:
            import quantus.full_workflow
            import quantus.data_objs.image
            import quantus.data_objs.seg
            import quantus.data_objs.analysis
            import quantus.data_objs.analysis_config
            import quantus.data_objs.visualizations
            import quantus.data_objs.data_export
            import quantus.image_loading.utc_loaders.transforms
            
            # Map Internal-TUL.QuantUS-QUS.full_workflow to quantus.full_workflow
            sys.modules["Internal-TUL.QuantUS-QUS.full_workflow"] = quantus.full_workflow
            
            # Map Internal-TUL.QuantUS-QUS.image_loading.transforms to quantus.image_loading.utc_loaders.transforms
            sys.modules["Internal-TUL.QuantUS-QUS.image_loading.transforms"] = quantus.image_loading.utc_loaders.transforms
            
            # Map Internal-TUL.data_objs to quantus.data_objs
            import quantus.data_objs
            sys.modules["Internal-TUL.data_objs"] = quantus.data_objs
            sys.modules["Internal-TUL.data_objs.image"] = quantus.data_objs.image
            sys.modules["Internal-TUL.data_objs.seg"] = quantus.data_objs.seg
            sys.modules["Internal-TUL.data_objs.analysis"] = quantus.data_objs.analysis
            sys.modules["Internal-TUL.data_objs.analysis_config"] = quantus.data_objs.analysis_config
            sys.modules["Internal-TUL.data_objs.visualizations"] = quantus.data_objs.visualizations
            sys.modules["Internal-TUL.data_objs.data_export"] = quantus.data_objs.data_export
        except ImportError as e:
            print(f"Warning: Could not inject dummy modules for Internal-TUL: {e}")

        for folder in internal_tul_path.iterdir():
            if folder.is_dir() and not folder.name.startswith("_"):
                try:
                    main_py = folder / "main.py"
                    if not main_py.exists():
                        continue
                    
                    # Use absolute module path from project root
                    rel_path = main_py.relative_to(project_root)
                    module_name = ".".join(rel_path.with_suffix("").parts)
                    
                    if module_name in sys.modules:
                        del sys.modules[module_name]
                    if folder.name in sys.modules:
                        del sys.modules[folder.name]
                        
                    module = importlib.import_module(module_name)
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

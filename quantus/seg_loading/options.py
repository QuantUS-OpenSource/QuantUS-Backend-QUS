from pathlib import Path

from argparse import ArgumentParser

from .functions import *

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
    import sys
    import importlib
    from pathlib import Path

    functions = {}

    # 1. Load from internal-TUL if available
    potential_project_roots = [
        Path(__file__).parents[4],
        Path(__file__).parents[5],
    ]
    
    project_root = None
    for root in potential_project_roots:
        if (root / "Internal-TUL").exists():
            project_root = root
            break
    
    if project_root is None:
        project_root = potential_project_roots[0]

    internal_tul_path = project_root / "Internal-TUL" / "QuantUS-QUS" / "processing"

    if internal_tul_path.exists():
        # Internal modules in QUS depend on quantus.full_workflow from engines/qus/quantus
        qus_engine_root = project_root / "engines" / "qus"
        if qus_engine_root.exists() and str(qus_engine_root) not in sys.path:
            sys.path.append(str(qus_engine_root))
            
        internal_tul_parent = project_root
        if internal_tul_parent.exists() and str(internal_tul_parent) not in sys.path:
            sys.path.append(str(internal_tul_parent))

        for item in internal_tul_path.iterdir():
            if item.is_file() and not item.name.startswith("_") and item.suffix == ".py":
                try:
                    rel_path = item.relative_to(project_root)
                    module_name = ".".join(rel_path.with_suffix("").parts)
                    
                    if module_name in sys.modules:
                        del sys.modules[module_name]
                    
                    full_wf_name = "Internal-TUL.QuantUS-QUS.full_workflow"
                    if full_wf_name not in sys.modules:
                        import quantus.full_workflow
                        sys.modules[full_wf_name] = quantus.full_workflow

                    module = importlib.import_module(module_name)
                    # For QUS seg loaders, we look for a dict with 'func' and 'exts'
                    # Or we look for functions decorated with @extensions
                    for name, obj in vars(module).items():
                        if isinstance(obj, dict) and 'func' in obj and 'exts' in obj:
                            functions[name] = {
                                'func': obj['func'],
                                'exts': obj['exts']
                            }
                except Exception as e:
                    print(f"Internal module {item.name} could not be loaded: {e}")

    # 2. Load from public functions
    for name, obj in globals().items():
        if type(obj) is dict:
            try:
                if callable(obj['func']) and obj['func'].__module__ == __package__ + '.functions':
                    functions[name] = {}
                    functions[name]['func'] = obj['func']
                    functions[name]['exts'] = obj['exts']
            except KeyError:
                pass
            
    return functions

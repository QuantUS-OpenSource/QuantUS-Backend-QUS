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

    internal_tul_path = project_root / "Internal-TUL" / "QuantUS-QUS" / "configs"

    if internal_tul_path.exists():
        # Internal modules in QUS depend on quantus.full_workflow from engines/qus/quantus
        qus_engine_root = project_root / "engines" / "qus"
        if qus_engine_root.exists() and str(qus_engine_root) not in sys.path:
            sys.path.append(str(qus_engine_root))
            
        # Add the parent of Internal-TUL to sys.path
        internal_tul_parent = project_root
        if internal_tul_parent.exists() and str(internal_tul_parent) not in sys.path:
            sys.path.append(str(internal_tul_parent))

        # Create a mock quantus package inside Internal-TUL to satisfy relative imports
        # if the internal modules are trying to do 'from ..full_workflow import ...'
        # while being loaded from Internal-TUL.QuantUS-QUS.configs.module
        internal_quantus_path = project_root / "Internal-TUL" / "QuantUS-QUS" / "quantus"
        if not internal_quantus_path.exists():
            try:
                internal_quantus_path.mkdir(parents=True, exist_ok=True)
                (internal_quantus_path / "__init__.py").touch()
                # Link full_workflow.py from engines/qus/quantus/full_workflow.py
                # Or just add engines/qus to sys.path and let the import system find it if we name it correctly.
            except Exception:
                pass

        for item in internal_tul_path.iterdir():
            if item.is_file() and not item.name.startswith("_") and item.suffix == ".py":
                try:
                    # Calculate module path relative to project_root
                    rel_path = item.relative_to(project_root)
                    module_name = ".".join(rel_path.with_suffix("").parts)
                    
                    if module_name in sys.modules:
                        del sys.modules[module_name]
                    
                    # We inject a dummy 'Internal-TUL.QuantUS-QUS.full_workflow' if it's missing
                    # since internal modules seem to expect it there.
                    full_wf_name = "Internal-TUL.QuantUS-QUS.full_workflow"
                    if full_wf_name not in sys.modules:
                        import quantus.full_workflow
                        sys.modules[full_wf_name] = quantus.full_workflow

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

"""Shared plugin-discovery and decorator machinery used by every stage's options.py/decorators.py.

Every stage supports two plugin sources, merged additively:
  1. Built-in, in-repo plugins (shipped with this package).
  2. External, dropped-in plugins under get_external_plugin_root() — used for
     private/proprietary plugins (an approved user drops a received plugin folder/file
     in, entirely outside this repo's git working tree) and for a future GUI's dynamic
     plugin loading. On a name collision the external plugin wins and a warning is
     emitted; external plugins never disable or replace the rest of the built-in
     public plugin set.

Plugin/file names only need to be unique within the directory discovery actually scans
(the same guarantee the filesystem already enforces for built-in plugins) — not
globally across all six stages.

See the per-stage README files for the plugin contract each stage expects.
"""

import importlib
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional


def get_quantus_home_dir() -> Path:
    """Root directory for QuantUS's local state (external plugins, default output
    locations, etc), keeping it out of the repo's tracked files by default.

    Always <repo_root>/.quantus/ (gitignored) — not itself configurable; individual
    locations under it (e.g. the external plugin root) may have their own more
    specific override. This assumes the package is run directly out of a repo
    checkout, as documented in CLAUDE.md, rather than installed elsewhere via pip.
    """
    return Path(__file__).resolve().parent.parent / ".quantus"


def get_external_plugin_root() -> Optional[Path]:
    """Root directory scanned for externally dropped-in plugins, additive to the
    built-in public plugin set in every stage.

    Configurable via the QUANTUS_PLUGIN_DIR env var; defaults to
    <repo_root>/.quantus/plugins/. Mirrors the in-repo package layout one level down,
    e.g. QUANTUS_PLUGIN_DIR/analysis/paramap/analysis_methods/my_method.py or
    QUANTUS_PLUGIN_DIR/image_loading/my_scanner/main.py.

    Returns:
        The root directory, or None if it doesn't exist (nothing to scan).
    """
    configured = os.environ.get("QUANTUS_PLUGIN_DIR")
    root = Path(configured).expanduser() if configured else get_quantus_home_dir() / "plugins"
    return root if root.is_dir() else None


def import_external_module(directory: Path, module_name: str):
    """Import `module_name` (which may be dotted, e.g. "my_scanner.main") as if
    `directory` were on sys.path, so a dropped-in plugin's own relative imports to
    sibling files (e.g. `from .parser import X` inside a folder-based plugin) resolve
    correctly, exactly as they would for a built-in plugin.

    Only imports relative to the plugin's own folder work this way. A relative import
    reaching further up (e.g. a built-in image loader's `from ....data_objs.image import X`,
    valid because it's nested several levels inside the real `quantus` package) will raise
    "attempted relative import beyond top-level package" once external, since externally
    the plugin is only ever nested one level deep. Plugins meant for the external
    directory must use absolute imports (`from quantus.data_objs.image import X`) for
    anything outside their own folder.
    """
    directory_str = str(directory)
    if directory_str not in sys.path:
        sys.path.insert(0, directory_str)
    return importlib.import_module(module_name)


def merge_external_plugins(discovered: dict, external: dict, source: Path) -> dict:
    """Merge externally discovered plugins into `discovered` additively, in place.

    On a name collision the external plugin wins and a warning is emitted (see module
    docstring for the rationale).
    """
    for name, entry in external.items():
        if name in discovered:
            warnings.warn(
                f"External plugin '{name}' in {source} overrides the built-in plugin of the same name."
            )
        discovered[name] = entry
    return discovered


def discover_plugin_classes(base_dir: Path, package: str, module_name: str,
                             class_name_fn: Callable[[str], str],
                             external_stage: Optional[str] = None) -> dict:
    """Scan immediate subfolders of base_dir for a `module_name` submodule and pull an
    object off it whose name is given by class_name_fn(folder_name).

    Args:
        base_dir: Directory whose immediate subfolders are candidate built-in plugins.
        package: Caller's __package__, used to resolve the relative import.
        module_name: Submodule to import from each plugin folder (e.g. "framework", "main").
        class_name_fn: Maps a folder name to the attribute name expected on that submodule.
        external_stage: If given, also scans get_external_plugin_root()/external_stage for
            the same shape of plugin (e.g. "analysis", "image_loading") and merges the
            results in additively (external wins on collision).

    Returns:
        dict: {folder_name: entry_object}.
    """
    discovered = {}
    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f".{folder.name}.{module_name}", package=package)
        except ModuleNotFoundError:
            continue
        entry = getattr(module, class_name_fn(folder.name), None)
        if entry is not None:
            discovered[folder.name] = entry

    if external_stage:
        external_root = get_external_plugin_root()
        external_dir = external_root / external_stage if external_root else None
        if external_dir and external_dir.is_dir():
            external = {}
            for folder in sorted(external_dir.iterdir()):
                if not folder.is_dir() or folder.name.startswith("_"):
                    continue
                if not (folder / f"{module_name}.py").is_file():
                    continue
                module = import_external_module(external_dir, f"{folder.name}.{module_name}")
                entry = getattr(module, class_name_fn(folder.name), None)
                if entry is not None:
                    external[folder.name] = entry
            merge_external_plugins(discovered, external, external_dir)

    return discovered


def _candidate_module_names(directory: Path) -> List[str]:
    """Immediate children of `directory` importable as a submodule: either a `.py` file
    (imported by its stem) or a folder containing `__init__.py` (imported by its folder
    name, as a package) -- the folder form lets a plugin bundle sibling data assets (e.g.
    a lookup table) alongside its code, loaded via a path relative to the plugin's own
    `__file__`. Anything starting with "_" is skipped either way.
    """
    names = []
    for entry in sorted(directory.iterdir()):
        if entry.name.startswith("_"):
            continue
        if entry.is_file() and entry.suffix == ".py":
            names.append(entry.stem)
        elif entry.is_dir() and (entry / "__init__.py").is_file():
            names.append(entry.name)
    return names


def discover_marked_functions(package: str, relative_module_prefix: str, funcs_dir: Path,
                               marker_attr: str, external_funcs_dir: Optional[Path] = None) -> dict:
    """Import every file in funcs_dir and collect module-level callables carrying marker_attr.

    Args:
        package: Caller's __package__, used to resolve the relative import.
        relative_module_prefix: Relative module path containing funcs_dir (e.g. ".paramap.analysis_methods").
        funcs_dir: Directory whose files are candidate built-in plugin functions.
        marker_attr: Attribute name (set by a decorator) that marks a callable as a discoverable plugin.
        external_funcs_dir: If given, also imports every file/folder-plugin here directly
            (not as a package submodule) and merges the results in additively (external
            wins on collision).

    Returns:
        dict: {name: callable}.
    """
    discovered = {}
    if funcs_dir.is_dir():
        for module_name in _candidate_module_names(funcs_dir):
            try:
                module = importlib.import_module(f"{relative_module_prefix}.{module_name}", package=package)
            except ModuleNotFoundError:
                continue
            for name, obj in vars(module).items():
                if callable(obj) and not isinstance(obj, type) and hasattr(obj, marker_attr):
                    discovered[name] = obj

    if external_funcs_dir and external_funcs_dir.is_dir():
        external = {}
        for module_name in _candidate_module_names(external_funcs_dir):
            module = import_external_module(external_funcs_dir, module_name)
            for name, obj in vars(module).items():
                if callable(obj) and not isinstance(obj, type) and hasattr(obj, marker_attr):
                    external[name] = obj
        merge_external_plugins(discovered, external, external_funcs_dir)

    return discovered


@dataclass
class PluginParam:
    """A single configurable keyword argument for a plugin, with its default (if any).

    Normalizes across the different name/default decorator-attribute pairs used by
    different stages (e.g. required_kwargs/default_kwarg_vals for analysis methods,
    gui_kwargs/default_gui_kwarg_vals for config loaders) into one stable shape a GUI
    can introspect to build dynamic form fields from, without needing to know which
    stage-specific attribute names apply to a given plugin.
    """
    name: str
    default: Any = None
    has_default: bool = False


def get_plugin_params(entry, names_attr: str, defaults_attr: Optional[str] = None) -> List[PluginParam]:
    """Normalize a plugin's kwargs/defaults decorator attributes into a list of PluginParam.

    Args:
        entry: The decorated plugin callable or class.
        names_attr: Attribute holding the kwarg names (e.g. "required_kwargs", "gui_kwargs").
        defaults_attr: Attribute holding the matching default values, if any (e.g.
            "default_kwarg_vals", "default_gui_kwarg_vals"). Matched to names_attr by
            position; if shorter than names_attr, the trailing names are left without a default.

    Returns:
        list: One PluginParam per name in names_attr, in order.
    """
    names = getattr(entry, names_attr, ()) or ()
    defaults = (getattr(entry, defaults_attr, ()) or ()) if defaults_attr else ()
    return [
        PluginParam(name=name, default=defaults[i] if i < len(defaults) else None, has_default=i < len(defaults))
        for i, name in enumerate(names)
    ]


def attr_decorator(attr_name: str) -> Callable:
    """Build a decorator factory that stores its variadic args as `attr_name` on the decorated callable.

    Replaces the near-identical `def decorator(func): func.X = args; return func` pattern
    repeated across every stage's decorators.py.
    """
    def decorator_factory(*args):
        def decorator(func):
            setattr(func, attr_name, args)
            return func
        return decorator
    return decorator_factory

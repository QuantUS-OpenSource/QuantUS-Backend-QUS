"""
Scans a user-writable directory for .py plugin files and loads them
at runtime using importlib to allow for correct functionality whether 
running from source or as a frozen PyInstaller bundle.

User plugin directories (created automatically on first launch):
  Windows:  %LOCALAPPDATA%\\QuantUS\\QuantUS\\plugins\\<category>\\
  macOS:    ~/Library/Application Support/QuantUS/plugins/<category>/
  Linux:    ~/.local/share/QuantUS/plugins/<category>/
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Any

try:
    from appdirs import user_data_dir           
    _BASE = Path(user_data_dir("QuantUS", "QuantUS"))
except ImportError:                              
    _BASE = Path.home() / ".quantus"

USER_PLUGINS_ROOT = _BASE / "plugins"


def get_user_plugin_dir(category: str) -> Path:
    """
    Return the user-writable plugin folder for category, creating it if absent.

    Call with the same names as the backend subpackage folders:
      'image_loading', 'seg_loading', 'analysis', 'visualizations', 'data_export'
    """
    d = USER_PLUGINS_ROOT / category
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_user_plugins(category: str) -> list[Any]:
    """
    Load every .py file in the user plugin folder for category.

    Returns a list of loaded module objects. Import failures emit a
    RuntimeWarning and are skipped.
    """
    plugin_dir = get_user_plugin_dir(category)
    loaded: list[Any] = []

    for py_file in sorted(plugin_dir.glob("*.py")):
        module_name = f"_quantus_user_{category}_{py_file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None or spec.loader is None:
                warnings.warn(
                    f"Could not create module spec for plugin: {py_file}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            loaded.append(module)
            print(f"[QuantUS] Loaded user plugin: {py_file.name} ({category})")
        except Exception as exc:
            warnings.warn(
                f"Failed to load user plugin '{py_file.name}': {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    return loaded
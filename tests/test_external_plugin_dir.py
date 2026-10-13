"""Smoke tests for the external plugin directory (QUANTUS_PLUGIN_DIR).

This is the mechanism private/proprietary plugins are meant to use: an approved user
drops a received plugin folder/file in here, entirely outside the git working tree, and
it shows up alongside the built-in public plugins next time discovery runs. It's also
the same mechanism a future GUI would use to load plugins dynamically.
"""

import textwrap
from pathlib import Path

import pytest

from quantus.image_loading.utc_loaders.options import get_scan_loaders
from quantus.analysis.options import get_analysis_types
from quantus.plugin_utils import get_quantus_home_dir


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))


def test_quantus_home_dir_is_repo_relative_not_user_home():
    repo_root = Path(__file__).resolve().parent.parent
    assert get_quantus_home_dir() == repo_root / ".quantus"


def test_external_root_absent_by_default(monkeypatch, tmp_path):
    import quantus.plugin_utils as plugin_utils

    monkeypatch.delenv("QUANTUS_PLUGIN_DIR", raising=False)
    monkeypatch.setattr(plugin_utils, "get_quantus_home_dir", lambda: tmp_path)  # no .quantus/plugins here
    assert plugin_utils.get_external_plugin_root() is None


def test_external_scan_loader_merged_additively(monkeypatch, tmp_path):
    _write(tmp_path / "image_loading" / "test_ext_scanner" / "main.py", """
        from quantus.data_objs.image import UltrasoundRfImage

        class EntryClass(UltrasoundRfImage):
            extensions = ['.testext']
            spatial_dims = 2
            gui_kwargs = []
            cli_kwargs = []
            default_gui_kwarg_vals = []
            default_cli_kwarg_vals = []
    """)
    monkeypatch.setenv("QUANTUS_PLUGIN_DIR", str(tmp_path))

    loaders = get_scan_loaders()
    assert "test_ext_scanner" in loaders
    assert "clarius_rf" in loaders  # built-ins still present alongside it
    assert loaders["test_ext_scanner"]["spatial_dims"] == 2


def test_external_analysis_method_collision_overrides_builtin(monkeypatch, tmp_path):
    _write(tmp_path / "analysis" / "paramap" / "analysis_methods" / "attenuation_coef.py", """
        from quantus.analysis.paramap.decorators import output_vars

        @output_vars("test_ext_override")
        def attenuation_coef(scan_rf_window, phantom_rf_window, window, config, image_data, **kwargs):
            pass
    """)
    monkeypatch.setenv("QUANTUS_PLUGIN_DIR", str(tmp_path))

    with pytest.warns(UserWarning, match="overrides the built-in plugin"):
        _, functions = get_analysis_types()
    assert functions["paramap"]["attenuation_coef"].outputs == ("test_ext_override",)


def test_external_analysis_method_folder_plugin_discovered(monkeypatch, tmp_path):
    """A plugin may also be a folder (with __init__.py) instead of a single .py file --
    lets it bundle sibling data assets (e.g. a lookup table), same shape as the built-in
    homodyned_k_params plugin (quantus/analysis/paramap/analysis_methods/homodyned_k_params/).
    """
    _write(tmp_path / "analysis" / "paramap" / "analysis_methods" / "test_ext_folder_method" / "__init__.py", """
        from quantus.analysis.paramap.decorators import output_vars

        @output_vars("test_ext_folder_output")
        def test_ext_folder_method(scan_rf_window, phantom_rf_window, window, config, image_data, **kwargs):
            pass
    """)
    monkeypatch.setenv("QUANTUS_PLUGIN_DIR", str(tmp_path))

    _, functions = get_analysis_types()
    assert "test_ext_folder_method" in functions["paramap"]
    assert functions["paramap"]["test_ext_folder_method"].outputs == ("test_ext_folder_output",)
    assert "attenuation_coef" in functions["paramap"]  # built-ins still present alongside it


def test_external_aggregate_plugin_discovered_from_sibling_folder(monkeypatch, tmp_path):
    """A @location("aggregate") plugin lives in its own sibling folder, analysis_aggregators/,
    not nested inside analysis_methods/ -- mirrors the built-in layout
    (quantus/analysis/paramap/analysis_aggregators/ac_scan_stats.py etc.) and must be
    discovered from there, merged in alongside window/full_segmentation plugins found in
    analysis_methods/.
    """
    _write(tmp_path / "analysis" / "paramap" / "analysis_aggregators" / "test_ext_aggregate_method.py", """
        from quantus.analysis.paramap.decorators import output_vars, location

        @output_vars("test_ext_aggregate_output")
        @location("aggregate")
        def test_ext_aggregate_method(windows, aggregate_results, config, image_data, **kwargs):
            pass
    """)
    monkeypatch.setenv("QUANTUS_PLUGIN_DIR", str(tmp_path))

    _, functions = get_analysis_types()
    assert "test_ext_aggregate_method" in functions["paramap"]
    assert functions["paramap"]["test_ext_aggregate_method"].outputs == ("test_ext_aggregate_output",)
    assert functions["paramap"]["test_ext_aggregate_method"].location == ("aggregate",)
    assert "attenuation_coef" in functions["paramap"]  # built-ins still present alongside it

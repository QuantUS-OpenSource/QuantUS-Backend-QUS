"""Smoke tests for PluginParam / get_plugin_params, the normalized kwargs/defaults
contract intended for a future GUI to introspect and build dynamic form fields from.
"""

from quantus.plugin_utils import get_plugin_params
from quantus.analysis.paramap.analysis_methods.attenuation_coef import attenuation_coef
from quantus.analysis.options import get_required_kwargs


def test_get_plugin_params_matches_required_kwargs_and_defaults():
    params = get_plugin_params(attenuation_coef, 'required_kwargs', 'default_kwarg_vals')
    assert [p.name for p in params] == list(attenuation_coef.required_kwargs)
    assert [p.default for p in params] == list(attenuation_coef.default_kwarg_vals)
    assert all(p.has_default for p in params)


def test_get_plugin_params_without_defaults_attr():
    params = get_plugin_params(attenuation_coef, 'required_kwargs')
    assert [p.name for p in params] == list(attenuation_coef.required_kwargs)
    assert all(p.default is None and not p.has_default for p in params)


def test_get_required_kwargs_uses_plugin_params_consistently():
    organized = get_required_kwargs('paramap', ['attenuation_coef'])
    params = {p.name: p.default for p in get_plugin_params(attenuation_coef, 'required_kwargs', 'default_kwarg_vals')}
    assert organized['attenuation_coef'] == params

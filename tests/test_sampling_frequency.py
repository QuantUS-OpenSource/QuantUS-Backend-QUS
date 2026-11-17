"""The scan file's sampling frequency is the truth (e.g. Clarius rf.yml 'sampling rate').

ParamapAnalysis fills it into a config that has none and rejects a config that disagrees;
clarius_sampling_frequency reads it from a capture's rf.yml without loading the data.
"""

import pytest

from quantus.analysis.paramap.framework import ParamapAnalysis
from quantus.image_loading.utc_loaders.clarius_rf.parser import clarius_sampling_frequency

from test_phantom_window import _make_inputs


def test_config_without_fs_takes_the_scan_files():
    image, seg, config = _make_inputs()
    image.sampling_frequency = 30e6
    config.sampling_frequency = None
    ParamapAnalysis(image, config, seg, [])
    assert config.sampling_frequency == 30e6


def test_config_fs_must_match_the_scan_files():
    image, seg, config = _make_inputs()
    image.sampling_frequency = 20e6
    config.sampling_frequency = 30e6
    with pytest.raises(ValueError, match="does not match the scan file's 20 MHz"):
        ParamapAnalysis(image, config, seg, [])
    config.sampling_frequency = 20e6
    ParamapAnalysis(image, config, seg, [])  # matching value is fine


def test_loader_without_fs_uses_config():
    image, seg, config = _make_inputs()
    image.sampling_frequency = None
    config.sampling_frequency = 30e6
    ParamapAnalysis(image, config, seg, [])
    config.sampling_frequency = None
    with pytest.raises(ValueError, match="Sampling frequency unknown"):
        ParamapAnalysis(image, config, seg, [])


def test_clarius_sampling_frequency_from_rf_yml(tmp_path):
    capture = tmp_path / "capture_1"
    capture.mkdir()
    (capture / "t_rf.raw").write_bytes(b"")
    (capture / "t_rf.yml").write_text("frames: 1\ntransmit frequency: 4.00 MHz\nsampling rate: 20 MHz\n")
    assert clarius_sampling_frequency(str(capture / "t_rf.raw")) == 20e6
    assert clarius_sampling_frequency(str(capture)) == 20e6
    tar_dir = tmp_path / "scan_extracted"
    tar_dir.mkdir()
    (tar_dir / "s_rf.yml").write_text("sampling rate: 15 MHz\n")
    assert clarius_sampling_frequency(str(tmp_path / "scan.tar")) == 15e6
    with pytest.raises(FileNotFoundError):
        clarius_sampling_frequency(str(tmp_path / "missing" / "x_rf.raw"))

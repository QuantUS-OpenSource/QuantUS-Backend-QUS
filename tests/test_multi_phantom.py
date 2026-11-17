"""Several reference frames / files: the reference spectrum is averaged over all of them.

Loaders put every reference frame in image_data.phantom_rf_frames (n_frames, samples, lines);
the "central_lines" and "matched_all_frames" phantom-window policies concatenate the same
slice from every frame, so line-averaged reference spectra average over frames too.
Synthetic RF only (no sample scans ship with this repo).
"""

import numpy as np
import pytest

from quantus.analysis.paramap.framework import ParamapAnalysis
from quantus.analysis.paramap.transforms import compute_power_spec
from pathlib import Path

from quantus.data_objs.image import UltrasoundRfImage, as_path_list, find_data_files
from quantus.power_spectrum import compare_spectra
from quantus.power_spectrum.spectrum import LOW_FREQ_IDX_START, roi_power_spectrum_db

from test_phantom_window import _make_inputs, _window


def _with_frames(image, n_frames=3, seed=1):
    rng = np.random.default_rng(seed)
    n_ax, n_lat = image.phantom_rf_data.shape
    image.phantom_rf_frames = rng.normal(size=(n_frames, n_ax, n_lat))
    image.phantom_rf_data = image.phantom_rf_frames[0]
    return image


def test_as_path_list_and_names():
    assert as_path_list("a.tar") == ["a.tar"]
    assert as_path_list(["a.tar", "b.tar"]) == ["a.tar", "b.tar"]
    with pytest.raises(ValueError):
        as_path_list([])
    image = UltrasoundRfImage("scan.tar", ["ref1.tar", "ref2.tar", "ref3.tar"])
    assert image.phantom_path == "ref1.tar"
    assert image.phantom_paths == ["ref1.tar", "ref2.tar", "ref3.tar"]
    assert image.phantom_name == "ref1 (+2 more)"
    assert UltrasoundRfImage("scan.tar", "ref1.tar").phantom_name == "ref1"


def test_get_phantom_frames_falls_back_to_single_frame():
    image, _, _ = _make_inputs()
    frames = image.get_phantom_frames()
    assert frames.shape == (1,) + image.phantom_rf_data.shape
    np.testing.assert_array_equal(frames[0], image.phantom_rf_data)


def test_single_frame_policies_unchanged():
    """One reference frame: all-frame policies give exactly the old single-frame slices."""
    image, seg, config = _make_inputs()
    analysis = ParamapAnalysis(image, config, seg, [])
    window = _window(100, 199, 10, 19)
    np.testing.assert_array_equal(analysis.get_phantom_window(window, "matched_all_frames"),
                                  analysis.get_phantom_window(window, "matched"))
    np.testing.assert_array_equal(analysis.get_phantom_window(window, "central_lines"),
                                  image.phantom_rf_data[100:200, 23:72])


def test_all_frame_policies_concatenate_every_frame():
    image, seg, config = _make_inputs(n_lat_phantom=96)
    _with_frames(image, n_frames=3)
    analysis = ParamapAnalysis(image, config, seg, [])
    window = _window(100, 199, 10, 19)

    matched_all = analysis.get_phantom_window(window, "matched_all_frames")
    expected = np.concatenate([f[100:200, 10:20] for f in image.phantom_rf_frames], axis=1)
    np.testing.assert_array_equal(matched_all, expected)

    central = analysis.get_phantom_window(window, "central_lines")
    expected = np.concatenate([f[100:200, 23:72] for f in image.phantom_rf_frames], axis=1)
    np.testing.assert_array_equal(central, expected)

    # "matched" stays first-frame only (hscan / bsc_stft divide element-wise)
    np.testing.assert_array_equal(analysis.get_phantom_window(window, "matched"),
                                  image.phantom_rf_frames[0][100:200, 10:20])


def test_concatenated_power_spectrum_is_mean_over_frames():
    """Line-averaged PS of concatenated frames == mean of the per-frame line-averaged PS."""
    image, seg, config = _make_inputs()
    _with_frames(image, n_frames=4)
    analysis = ParamapAnalysis(image, config, seg, [])
    window = _window(100, 227, 10, 19)
    args = (config.transducer_freq_band[0], config.transducer_freq_band[1], config.sampling_frequency, 256)
    _, ps_all = compute_power_spec(analysis.get_phantom_window(window, "matched_all_frames"), *args)
    per_frame = [compute_power_spec(f[100:228, 10:20], *args)[1] for f in image.phantom_rf_frames]
    np.testing.assert_allclose(ps_all, np.mean(per_frame, axis=0), rtol=1e-10)


def test_compute_power_spectra_uses_all_reference_frames():
    image, seg, config = _make_inputs()
    _with_frames(image, n_frames=2)
    analysis = ParamapAnalysis(image, config, seg, ["compute_power_spectra"], n_fft=256)
    analysis.compute_paramaps()
    w = analysis.windows[0]
    args = (config.transducer_freq_band[0], config.transducer_freq_band[1], config.sampling_frequency, 256)
    sl = (slice(w.ax_min, w.ax_max + 1), slice(w.lat_min, w.lat_max + 1))
    per_frame = [compute_power_spec(f[sl], *args)[1] for f in image.phantom_rf_frames]
    np.testing.assert_allclose(w.results.r_ps, 20 * np.log10(np.mean(per_frame, axis=0)), rtol=1e-10)


def test_roi_spectrum_of_stack_averages_frames():
    rng = np.random.default_rng(3)
    stack = rng.normal(size=(3, 400, 40))
    roi, n_fft = (50, 350, 10, 30), 1024
    _, p_stack = roi_power_spectrum_db(stack, 30e6, roi, n_fft=n_fft)
    power = np.mean([np.abs(np.fft.fft(fr[50:350, 10:30], n=n_fft, axis=0)) ** 2 for fr in stack], axis=(0, 2))
    expected = 10 * np.log10(power[LOW_FREQ_IDX_START: n_fft // 2])
    np.testing.assert_allclose(p_stack, expected - expected.max(), atol=1e-9)
    # a stack of identical frames equals the single 2D frame
    same = np.repeat(stack[:1], 3, axis=0)
    np.testing.assert_allclose(roi_power_spectrum_db(same, 30e6, roi, n_fft)[1],
                               roi_power_spectrum_db(stack[0], 30e6, roi, n_fft)[1], atol=1e-9)
    assert compare_spectra(stack[0], stack, 30e6).reference_db.shape == compare_spectra(stack[0], stack[0], 30e6).reference_db.shape


class _FolderLoader(UltrasoundRfImage):
    folder_patterns = ["*rf.raw", "*.tar"]


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_phantom_folder_expands_to_every_capture(tmp_path):
    ref = tmp_path / "Reference_L1Phantom"
    for i in (1, 2, 10):
        _touch(ref / f"capture_{i}" / f"t{i}_rf.raw")
        _touch(ref / f"capture_{i}" / f"t{i}_env.raw")  # not RF -> ignored
    image = _FolderLoader(str(tmp_path / "scan_rf.raw"), str(ref))
    assert [Path(p).parent.name for p in image.phantom_paths] == ["capture_1", "capture_2", "capture_10"]
    assert image.phantom_name == "Reference_L1Phantom (3 files)"
    assert image.phantom_path == image.phantom_paths[0]


def test_phantom_folder_tar_not_double_counted_with_its_extraction(tmp_path):
    _touch(tmp_path / "capture_1.tar")
    _touch(tmp_path / "capture_1_extracted" / "a_rf.raw")  # created by extracting capture_1.tar
    _touch(tmp_path / "capture_2_extracted" / "b_rf.raw")  # no tar next to it -> kept
    files = find_data_files(tmp_path, _FolderLoader.folder_patterns)
    assert [Path(f).name for f in files] == ["capture_1.tar", "b_rf.raw"]


def test_phantom_folder_errors(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="No reference data files"):
        _FolderLoader("scan.raw", str(tmp_path / "empty"))
    with pytest.raises(ValueError, match="does not accept a folder"):
        UltrasoundRfImage("scan.raw", str(tmp_path))
    f = _touch(tmp_path / "c" / "x_rf.raw")
    with pytest.raises(ValueError, match="more than once"):
        _FolderLoader("scan.raw", [str(tmp_path / "c"), str(f)])

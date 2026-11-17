# Standard Library Imports
from __future__ import annotations

# Third-Party Library Imports
import os
from pathlib import Path

# Local Module Imports
from ....data_objs.image import UltrasoundRfImage
import numpy as np

from .parser import ClariusTarUnpacker, clariusRfParserWrapper, clariusRfPhantomParser

class EntryClass(UltrasoundRfImage):
    """
    Class for Clarius RF image data.
    This class is used to parse RF data from Clarius ultrasound machines.
    Supports both raw RF files and compressed .tar files.
    """
    extensions = [".raw", ".tar"]  # Clarius RF files can be raw or tar compressed
    folder_patterns = ["*rf.raw", "*.tar"]  # a phantom folder: every capture (one rf.raw per capture folder, or .tar)
    spatial_dims = 2  # Clarius data can be 3D but we extract 2D frames
    gui_kwargs = ['use_tgc', 'start_depth_factor']; cli_kwargs = ['visualize']
    default_gui_kwarg_vals = ['True', 'None']  # str because parsed from GUI
    default_cli_kwarg_vals = ['False']  # str because parsed from CLI

    def __init__(self, scan_path: str, phantom_path: str, **kwargs):
        """Initialize the Clarius RF loader.
        
        Args:
            scan_path (str): Path to the scan RF file (.raw or .tar)
            phantom_path (str | list[str]): Reference phantom RF file(s) (.raw or .tar) and/or folder(s);
                a folder is searched recursively for every capture (*rf.raw, *.tar).
                Every frame of every file goes into phantom_rf_frames (averaged reference spectrum).
            **kwargs:
                visualize (bool): Whether to visualize the data in the terminal after extraction.
                use_tgc (bool): Whether to use TGC data in processing.
                start_depth_factor (float): Factor to multiply endDepth by for startDepth. Value of None uses default calculation.
        """
        super().__init__(scan_path, phantom_path)

        visualize = kwargs.get('visualize', False)
        use_tgc = kwargs.get('use_tgc', True)
        start_depth_factor = kwargs.get('start_depth_factor', None)

        # Validate file extensions
        scan_path = Path(scan_path)
        assert scan_path.suffix in self.extensions, f"Scan file must end with {self.extensions}"
        for p in self.phantom_paths:
            assert Path(p).suffix in self.extensions, f"Phantom file must end with {self.extensions}: {p}"

        # Handle tar files if provided and get extracted folder paths
        scan_raw_dir = self._handle_tar_file(str(scan_path))
        phantom_raw_dirs = [self._handle_tar_file(p) for p in self.phantom_paths]
        phantom_raw_dir = phantom_raw_dirs[0]
                
        # Parse the data using the Clarius parser
        imgData, imgInfo, refData, refInfo, scanConverted = clariusRfParserWrapper(
            os.path.dirname(scan_raw_dir),
            os.path.dirname(phantom_raw_dir),
            visualize=visualize,
            use_tgc=use_tgc,
            start_depth_factor=start_depth_factor
        )
        
        # Set the required attributes
        frame_idx = kwargs.get('ref_frame_idx', 0)
        
        # Extract the frame and ensure correct orientation
        self.rf_data = imgData.rf  # Shape will be (frames, axial, lateral)
        self.phantom_rf_data = refData.rf[frame_idx]

        # Sampling frequency from the scan's rf.yml (Hz); every reference capture must match it.
        self.sampling_frequency = float(imgInfo.samplingFrequency)
        self._check_phantom_sampling_frequency(self.phantom_paths[0], refInfo.samplingFrequency)

        # Every frame of every reference file, for the averaged reference spectrum.
        ref_frames = [np.asarray(refData.rf)]
        for p, raw_dir in zip(self.phantom_paths[1:], phantom_raw_dirs[1:]):
            extra_data, extra_info = clariusRfPhantomParser(os.path.dirname(raw_dir), visualize=visualize,
                                                            use_tgc=use_tgc, start_depth_factor=start_depth_factor)
            self._check_phantom_sampling_frequency(p, extra_info.samplingFrequency)
            extra = np.asarray(extra_data.rf)
            if extra.shape[1:] != ref_frames[0].shape[1:]:
                raise ValueError(f"Reference phantom {Path(p).name} has frames of shape {extra.shape[1:]}, but "
                                 f"{Path(self.phantom_paths[0]).name} has {ref_frames[0].shape[1:]}; "
                                 "all reference files must use the same probe and depth settings.")
            ref_frames.append(extra)
        self.phantom_rf_frames = np.concatenate(ref_frames, axis=0)  # (n_frames, samples, lines)
        self.bmode = imgData.bMode
        
        # Calculate resolutions
        self.axial_res = imgInfo.yResRF  # Use the RF resolution directly
        self.lateral_res = imgInfo.xResRF
        
        # Scan conversion data if necessary
        if scanConverted:
            self.sc_bmode = imgData.scBmode if imgData.scBmode is not None else None
            self.sc_axial_res = imgInfo.axialRes
            self.sc_lateral_res = imgInfo.lateralRes
            self.xmap = imgData.scBmodeStruct.xmap
            self.ymap = imgData.scBmodeStruct.ymap
            self.tilt = imgInfo.tilt1
            self.width = imgInfo.width1
            self.start_depth = imgInfo.startDepth1
            self.end_depth = imgInfo.endDepth1
            
    def _check_phantom_sampling_frequency(self, phantom_path: str, phantom_fs: float) -> None:
        if not np.isclose(phantom_fs, self.sampling_frequency, rtol=1e-3):
            raise ValueError(f"Reference phantom {Path(phantom_path).name} was sampled at {phantom_fs / 1e6:g} MHz "
                             f"but the scan at {self.sampling_frequency / 1e6:g} MHz (from rf.yml); "
                             "acquire the phantom with the same probe and settings.")

    def _handle_tar_file(self, file_path: str) -> str:
        """Handle tar file extraction if needed.
        
        Args:
            file_path (str): Path to the file to check
            
        Returns:
            str: Path to the raw file (either original path if raw or extracted path if tar)
        """
        if Path(file_path).suffix == '.tar':
            # Extract the tar file
            unpacker = ClariusTarUnpacker(file_path, extraction_mode='single_tar')
            
            # Get the extracted folder path
            extracted_folder = os.path.join(
                os.path.dirname(file_path),
                f"{Path(file_path).stem}_extracted"
            )
            
            # Find the rf.raw file in the extracted folder
            for root, _, files in os.walk(extracted_folder):
                for file in files:
                    if file.endswith('rf.raw'):
                        return os.path.join(root, file)
            
            raise FileNotFoundError(f"No rf.raw file found in extracted folder: {extracted_folder}")
        
        return file_path
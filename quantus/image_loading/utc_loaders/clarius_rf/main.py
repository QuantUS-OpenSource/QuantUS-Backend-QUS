# Standard Library Imports
from __future__ import annotations

# Third-Party Library Imports
import os
import numpy as np
from pathlib import Path

# Optional DICOM import for overlay functionality
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

# Local Module Imports
from ....data_objs.image import UltrasoundRfImage
from .parser import ClariusTarUnpacker, clariusRfParserWrapper

class EntryClass(UltrasoundRfImage):
    """
    Class for Clarius RF image data.
    This class is used to parse RF data from Clarius ultrasound machines.
    Supports both raw RF files and compressed .tar files.
    """
    extensions = [".raw", ".tar"]  # Clarius RF files can be raw or tar compressed
    spatial_dims = 2  # Clarius data can be 3D but we extract 2D frames
    gui_kwargs = ['use_tgc']; cli_kwargs = ['visualize']
    default_gui_kwarg_vals = ['True']  # str because parsed from GUI
    default_cli_kwarg_vals = ['False']  # str because parsed from CLI

    def __init__(self, scan_path: str, phantom_path: str, **kwargs):
        """Initialize the Clarius RF loader.
        
        Args:
            scan_path (str): Path to the scan RF file (.raw or .tar)
            phantom_path (str): Path to the phantom RF file (.raw or .tar)
            **kwargs:
                visualize (bool): Whether to visualize the data in the terminal after extraction.
                use_tgc (bool): Whether to use TGC data in processing.
        """
        super().__init__(scan_path, phantom_path)

        visualize = kwargs.get('visualize', False)
        use_tgc = kwargs.get('use_tgc', True)

        # Validate file extensions
        scan_path = Path(scan_path)
        phantom_path = Path(phantom_path)
        assert scan_path.suffix in self.extensions, f"Scan file must end with {self.extensions}"
        assert phantom_path.suffix in self.extensions, f"Phantom file must end with {self.extensions}"
        
        # Handle tar files if provided and get extracted folder paths
        scan_raw_dir = self._handle_tar_file(str(scan_path))
        phantom_raw_dir = self._handle_tar_file(str(phantom_path))
                
        # Parse the data using the Clarius parser
        imgData, imgInfo, refData, refInfo, scanConverted = clariusRfParserWrapper(
            os.path.dirname(scan_raw_dir),
            os.path.dirname(phantom_raw_dir),
            visualize=visualize,
            use_tgc=use_tgc,
        )
        
        # Set the required attributes
        frame_idx = kwargs.get('ref_frame_idx', 0)
        
        # Extract the frame and ensure correct orientation
        self.rf_data = imgData.rf  # Shape will be (frames, axial, lateral)
        self.phantom_rf_data = refData.rf[frame_idx]
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
        
        # Attempt to load DICOM image for overlay functionality
        if PYDICOM_AVAILABLE:
            self._load_dicom_overlay(scan_path)
        else:
            self.dicom_available = False
            
        # Attempt to load DICOM image for overlay functionality
        if PYDICOM_AVAILABLE:
            self._load_dicom_overlay(scan_path)
        else:
            self.dicom_available = False

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
    
    def _load_dicom_overlay(self, scan_path: str) -> None:
        """
        Attempt to load DICOM image for overlay functionality.
        
        This method looks for DICOM files in the same directory as the RF data
        and loads them for overlay display. If no DICOM files are found,
        the overlay functionality remains disabled.
        
        Args:
            scan_path (str): Path to the scan file (RF data)
        """
        if not PYDICOM_AVAILABLE:
            self.dicom_available = False
            return
            
        try:
            # Get the directory containing the RF data
            scan_dir = Path(scan_path).parent
            
            # Look for DICOM files in the same directory
            dicom_files = []
            for file_path in scan_dir.glob("*"):
                if file_path.is_file() and self._is_dicom_file(file_path):
                    dicom_files.append(file_path)
            
            if not dicom_files:
                # No DICOM files found, overlay not available
                self.dicom_available = False
                return
            
            # Load the first DICOM file found
            dicom_file = dicom_files[0]
            dicom_data = pydicom.dcmread(str(dicom_file))
            
            # Extract pixel data and ensure it's in the right format
            if hasattr(dicom_data, 'pixel_array'):
                dicom_pixels = dicom_data.pixel_array
                
                # Convert to grayscale if needed (handle different DICOM formats)
                if len(dicom_pixels.shape) == 3:
                    # RGB or multi-frame DICOM - convert to grayscale
                    if dicom_pixels.shape[2] == 3:  # RGB
                        dicom_pixels = np.dot(dicom_pixels[...,:3], [0.2989, 0.5870, 0.1140])
                    elif dicom_pixels.shape[0] < dicom_pixels.shape[2]:  # Multi-frame
                        dicom_pixels = dicom_pixels[0]  # Take first frame
                
                # Ensure the DICOM image matches the B-mode dimensions
                if self.sc_bmode is not None:
                    target_shape = self.sc_bmode.shape[:2] if len(self.sc_bmode.shape) > 2 else self.sc_bmode.shape
                else:
                    target_shape = self.bmode.shape[:2] if len(self.bmode.shape) > 2 else self.bmode.shape
                
                # Resize DICOM image to match B-mode dimensions if needed
                if dicom_pixels.shape[:2] != target_shape:
                    from skimage.transform import resize
                    dicom_pixels = resize(dicom_pixels, target_shape, preserve_range=True, anti_aliasing=True)
                
                # Normalize to 0-255 range and convert to uint8
                if dicom_pixels.dtype != np.uint8:
                    dicom_pixels = ((dicom_pixels - dicom_pixels.min()) / 
                                  (dicom_pixels.max() - dicom_pixels.min()) * 255).astype(np.uint8)
                
                self.dicom_image = dicom_pixels
                self.dicom_available = True
                
            else:
                self.dicom_available = False
                
        except Exception as e:
            # If any error occurs during DICOM loading, disable overlay
            print(f"Warning: Failed to load DICOM overlay: {e}")
            self.dicom_available = False
    
    def _is_dicom_file(self, file_path: Path) -> bool:
        """
        Check if a file is a DICOM file.
        
        Args:
            file_path (Path): Path to the file to check
            
        Returns:
            bool: True if the file is a DICOM file, False otherwise
        """
        if not PYDICOM_AVAILABLE:
            return False
            
        try:
            # Try to read the file as DICOM - if it succeeds, it's likely a DICOM file
            pydicom.dcmread(str(file_path), stop_before_pixels=True)
            return True
        except:
            return False

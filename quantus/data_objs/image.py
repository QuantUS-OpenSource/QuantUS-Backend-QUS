from pathlib import Path

import numpy as np

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

class UltrasoundRfImage:
    """
    Class for ultrasound RF image data.
    """

    def __init__(self, scan_path: str, phantom_path: str):
        # RF data
        self.rf_data: np.ndarray
        self.phantom_rf_data: np.ndarray
        self.bmode: np.ndarray
        self.axial_res: float # mm/pix
        self.lateral_res: float # mm/pix
        self.scan_name = Path(scan_path).stem
        self.phantom_name = Path(phantom_path).stem
        self.scan_path = scan_path
        self.phantom_path = phantom_path
        self.spatial_dims: int
        
        # Scan conversion parameters
        self.sc_bmode: np.ndarray = None
        self.xmap: np.ndarray # sc (y,x) --> preSC x
        self.ymap: np.ndarray # sc (y,x) --> preSC y
        self.width: float # deg
        self.tilt: float
        self.start_depth: float # mm
        self.end_depth: float # mm
        self.sc_axial_res: float # mm
        self.sc_lateral_res: float
        
        # 3D scan parameters
        self.coronal_res: float = None # mm/pix
        self.depth: float # depth in mm
        
        # 3D scan conversion parameters
        self.sc_bmode: np.ndarray = None
        self.coord_map_3d: np.ndarray # maps (z,y,x) in SC coords to (x,y) preSC coord
        self.sc_axial_res: float # mm/pix
        self.sc_lateral_res: float # mm/pix
        self.sc_coronal_res: float # mm/pix
        self.sc_params_3d = None
        
        # DICOM overlay parameters
        self.dicom_image: np.ndarray = None  # Original DICOM pixel data for overlay
        self.dicom_available: bool = False   # Flag indicating if DICOM overlay is available
        self.dicom_file_path: str = None     # Path to the loaded DICOM file

    def load_dicom_file(self, dicom_file_path: str) -> bool:
        """
        Manually load a DICOM file for overlay functionality.
        
        Args:
            dicom_file_path (str): Path to the DICOM file to load
            
        Returns:
            bool: True if DICOM was loaded successfully, False otherwise
        """
        if not PYDICOM_AVAILABLE:
            print("pydicom is not installed. Cannot load DICOM files.")
            return False
            
        try:
            dicom_path = Path(dicom_file_path)
            if not dicom_path.exists() or not dicom_path.is_file():
                print(f"DICOM file not found: {dicom_file_path}")
                return False
                
            # Read the DICOM file
            dicom_data = pydicom.dcmread(str(dicom_path))
            
            # Extract pixel data
            if hasattr(dicom_data, 'pixel_array'):
                dicom_pixels = dicom_data.pixel_array
                
                # Convert to grayscale if needed (handle different DICOM formats)
                if len(dicom_pixels.shape) == 4:
                    # 4D DICOM (frames, height, width, channels) - take first frame
                    dicom_pixels = dicom_pixels[0]
                if len(dicom_pixels.shape) == 3:
                    # RGB or multi-frame DICOM - convert to grayscale
                    if dicom_pixels.shape[2] == 3:  # RGB
                        dicom_pixels = np.dot(dicom_pixels[...,:3], [0.2989, 0.5870, 0.1140])
                    elif dicom_pixels.shape[0] < dicom_pixels.shape[2]:  # Multi-frame
                        dicom_pixels = dicom_pixels[0]  # Take first frame
                
                # Normalize to 0-255 range
                if dicom_pixels.dtype != np.uint8:
                    dicom_pixels = ((dicom_pixels - dicom_pixels.min()) / 
                                  (dicom_pixels.max() - dicom_pixels.min()) * 255).astype(np.uint8)
                
                # Crop black regions from top and bottom
                dicom_pixels = self._crop_black_regions(dicom_pixels)
                
                # Store the DICOM data
                self.dicom_image = dicom_pixels
                self.dicom_file_path = str(dicom_path)
                self.dicom_available = True
                
                print(f"Successfully loaded DICOM file: {dicom_path}")
                return True
            else:
                print("No pixel data found in DICOM file")
                return False
                
        except Exception as e:
            print(f"Failed to load DICOM file: {e}")
            self.dicom_available = False
            self.dicom_image = None
            self.dicom_file_path = None
            return False

    def _crop_black_regions(self, image: np.ndarray) -> np.ndarray:
        """
        Crop a fixed number of rows from top and bottom of the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Cropped image with specified rows removed
        """
        # Define fixed number of rows to crop from top and bottom
        # You can adjust these values as needed
        crop_top = 175   # Number of rows to crop from top
        crop_bottom = 175  # Number of rows to crop from bottom
        
        # Ensure we don't crop more than the image height
        height = image.shape[0]
        if crop_top + crop_bottom >= height:
            # If we try to crop too much, crop half from each side
            crop_top = crop_bottom = height // 4
        
        # Crop the image
        cropped_image = image[crop_top:height - crop_bottom, :]
        
        print(f"DICOM cropped: {image.shape} -> {cropped_image.shape} (removed {crop_top} from top, {crop_bottom} from bottom)")
        return cropped_image

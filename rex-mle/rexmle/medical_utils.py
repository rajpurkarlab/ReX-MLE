"""
Medical imaging utilities for loading, converting, and saving medical images.

All images are standardized to [D, H, W] format as per data_format_spec.md.
"""

import nibabel as nib
import pydicom
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Try to import SimpleITK (optional, for MHA/MHD/NRRD support)
try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False
    logger.warning("SimpleITK not installed. MHA/MHD/NRRD support disabled. Install with: pip install SimpleITK")

# Try to import nrrd (optional, alternative for NRRD support)
try:
    import nrrd
    HAS_NRRD = True
except ImportError:
    HAS_NRRD = False


def load_medical_image(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load medical image in any format and return in [D, H, W] format.

    Follows docs/specifications/data_format_spec.md specification.

    Supported formats:
    - DICOM (.dcm)
    - NIfTI (.nii, .nii.gz)
    - MHA/MHD (.mha, .mhd) - requires SimpleITK
    - NRRD (.nrrd, .seg.nrrd) - requires SimpleITK or pynrrd

    Args:
        filepath: Path to medical image

    Returns:
        Dictionary with:
        - 'image': numpy array in [D, H, W] format
        - 'metadata': dict with spacing, shape, dtype, etc.
        - 'source_format': str indicating original format

    Raises:
        ValueError: If file format not supported
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = filepath.suffix.lower()
    # Handle .seg.nrrd files (check full suffix for .seg.nrrd)
    suffixes = ''.join(filepath.suffixes).lower()

    if suffix == '.dcm' or (suffix == '' and _is_dicom_file(filepath)):
        return _load_dicom(filepath)
    elif suffix in ['.nii', '.gz']:
        return _load_nifti(filepath)
    elif suffix in ['.mha', '.mhd']:
        if not HAS_SITK:
            raise ImportError("SimpleITK required for MHA/MHD files. Install with: pip install SimpleITK")
        return _load_mha(filepath)
    elif suffix == '.nrrd' or suffixes.endswith('.seg.nrrd'):
        return _load_nrrd(filepath)
    else:
        raise ValueError(f"Unsupported format: {suffix}. Supported: .dcm, .nii, .nii.gz, .mha, .mhd, .nrrd, .seg.nrrd")


def _is_dicom_file(filepath: Path) -> bool:
    """Check if a file is DICOM by reading magic bytes."""
    try:
        with open(filepath, 'rb') as f:
            f.seek(128)
            magic = f.read(4)
            return magic == b'DICM'
    except:
        return False


def _fix_malformed_nrrd(filepath: Path) -> None:
    """
    Fix malformed NRRD files that are missing blank line between header and data.

    NRRD format requires a blank line separating the header from binary data.
    Some writers may omit this, causing SimpleITK to fail reading the file.

    Args:
        filepath: Path to NRRD file to fix in-place
    """
    import tempfile
    import shutil

    # Read the file in binary mode
    with open(filepath, 'rb') as f:
        content = f.read()

    # Find the end of the header (look for the last header field)
    # NRRD header fields end with a newline, and data starts after a blank line
    # We need to find where the header should end
    lines = content.split(b'\n')

    header_lines = []
    data_start_idx = -1

    for i, line in enumerate(lines):
        # Header lines contain ": " or start with "#"
        if b': ' in line or line.startswith(b'#') or line.startswith(b'NRRD'):
            header_lines.append(line)
        elif line.strip() == b'':
            # Empty line marks end of header
            data_start_idx = i + 1
            break
        else:
            # This is likely where the data starts (no blank line was present)
            data_start_idx = i
            break

    if data_start_idx == -1:
        # Couldn't find data start, leave file unchanged
        return

    # Reconstruct the file with proper blank line
    # Create temp file in system temp dir (not filepath.parent which may be read-only)
    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as tmp:
        tmp_path = Path(tmp.name)

        # Write header lines
        for line in header_lines:
            tmp.write(line + b'\n')

        # Write blank line separator
        tmp.write(b'\n')

        # Write the rest (data portion)
        data_portion = b'\n'.join(lines[data_start_idx:])
        tmp.write(data_portion)

    # Replace original file with fixed version
    try:
        shutil.move(str(tmp_path), str(filepath))
        logger.info(f"Fixed malformed NRRD file: {filepath}")
    except Exception as e:
        # If we can't write to the original location, clean up temp file
        logger.error(f"Failed to fix NRRD file {filepath}: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _load_dicom(filepath: Path) -> Dict[str, Any]:
    """
    Load DICOM file and return in [D, H, W] format.

    Args:
        filepath: Path to DICOM file

    Returns:
        Image data dictionary
    """
    dcm = pydicom.dcmread(str(filepath))

    # Get pixel array (original data)
    image = dcm.pixel_array.copy()
    original_dtype = str(image.dtype)

    # Convert to [D, H, W]
    if len(image.shape) == 2:
        # Single slice: [H, W] -> [1, H, W]
        image = np.expand_dims(image, axis=0)
    elif len(image.shape) == 3:
        # Multi-slice: already [D, H, W] in most DICOM
        pass

    # Apply rescale slope/intercept for CT (converts stored values to Hounsfield Units)
    if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
        rescale_slope = float(dcm.RescaleSlope)
        rescale_intercept = float(dcm.RescaleIntercept)
        if rescale_slope != 1.0 or rescale_intercept != 0.0:
            image = (image.astype(np.float64) * rescale_slope + rescale_intercept).astype(np.int16)

    # Extract spacing
    pixel_spacing = getattr(dcm, 'PixelSpacing', [1.0, 1.0])
    slice_thickness = float(getattr(dcm, 'SliceThickness', 1.0))
    spacing = [slice_thickness, float(pixel_spacing[0]), float(pixel_spacing[1])]

    # Build metadata
    metadata = {
        'spacing': spacing,
        'shape': list(image.shape),
        'dtype': str(image.dtype),
        'original_dtype': original_dtype,
        'modality': str(dcm.get('Modality', 'UNKNOWN')),
        'patient_id': str(dcm.get('PatientID', 'UNKNOWN'))
    }

    return {
        'image': image,
        'metadata': metadata,
        'source_format': 'dicom'
    }


def _load_nifti(filepath: Path) -> Dict[str, Any]:
    """
    Load NIfTI file and return in [D, H, W] format.

    NIfTI convention: (X, Y, Z) -> [D, H, W] = [Z, X, Y]

    Args:
        filepath: Path to NIfTI file

    Returns:
        Image data dictionary
    """
    nii = nib.load(str(filepath))

    # Get original data with its original dtype
    image = np.asanyarray(nii.dataobj)
    original_dtype = str(image.dtype)

    # Convert to [D, H, W] format
    if len(image.shape) == 2:
        # 2D image: [X, Y] -> [1, X, Y]
        image = np.expand_dims(image, axis=0)
    elif len(image.shape) == 3:
        # 3D volume: NIfTI is (X, Y, Z), convert to [D, H, W] = [Z, X, Y]
        image = np.transpose(image, (2, 0, 1))
    elif len(image.shape) == 4:
        # 4D NIfTI (e.g., fMRI time series), take first volume
        image = np.transpose(image[..., 0], (2, 0, 1))
    else:
        raise ValueError(f"Unsupported NIfTI shape: {image.shape}")

    # Get spacing from header (voxel dimensions)
    zooms = nii.header.get_zooms()
    if len(zooms) >= 3:
        # NIfTI zooms are (X, Y, Z), convert to [D, H, W] = [Z, X, Y]
        spacing = [float(zooms[2]), float(zooms[0]), float(zooms[1])]
    elif len(zooms) == 2:
        spacing = [1.0, float(zooms[0]), float(zooms[1])]
    else:
        spacing = [1.0, 1.0, 1.0]

    # Build metadata
    metadata = {
        'spacing': spacing,
        'shape': list(image.shape),
        'dtype': str(image.dtype),
        'original_dtype': original_dtype
    }

    return {
        'image': image,
        'metadata': metadata,
        'source_format': 'nifti'
    }


def _load_mha(filepath: Path) -> Dict[str, Any]:
    """
    Load MHA/MHD file using SimpleITK and return in [D, H, W] format.

    Args:
        filepath: Path to MHA/MHD file

    Returns:
        Image data dictionary
    """
    sitk_image = sitk.ReadImage(str(filepath))
    image = sitk.GetArrayFromImage(sitk_image)

    # SimpleITK returns [D, H, W] already
    original_dtype = str(image.dtype)

    # Get spacing (SimpleITK returns [X, Y, Z], reverse to [Z, X, Y] = [D, H, W])
    spacing_sitk = sitk_image.GetSpacing()
    spacing = [float(spacing_sitk[2]), float(spacing_sitk[0]), float(spacing_sitk[1])] if len(spacing_sitk) >= 3 else [1.0, 1.0, 1.0]

    # Build metadata
    metadata = {
        'spacing': spacing,
        'shape': list(image.shape),
        'dtype': str(image.dtype),
        'original_dtype': original_dtype,
        'origin': list(sitk_image.GetOrigin()),
        'direction': list(sitk_image.GetDirection())
    }

    return {
        'image': image,
        'metadata': metadata,
        'source_format': 'mha'
    }


def _load_nrrd(filepath: Path) -> Dict[str, Any]:
    """
    Load NRRD file and return in [D, H, W] format.

    Supports both .nrrd and .seg.nrrd files.
    Uses SimpleITK if available, otherwise falls back to pynrrd library.

    Args:
        filepath: Path to NRRD file

    Returns:
        Image data dictionary
    """
    # Prefer SimpleITK as it handles spacing/orientation consistently
    if HAS_SITK:
        try:
            sitk_image = sitk.ReadImage(str(filepath))
            image = sitk.GetArrayFromImage(sitk_image)
        except RuntimeError as e:
            # Try to fix malformed NRRD files (missing blank line between header and data)
            error_msg = str(e)
            if "didn't see" in error_msg or "NrrdImageIO" in error_msg:
                logger.warning(f"Malformed NRRD file detected, attempting to fix: {filepath}")
                try:
                    _fix_malformed_nrrd(filepath)
                    sitk_image = sitk.ReadImage(str(filepath))
                    image = sitk.GetArrayFromImage(sitk_image)
                except Exception as fix_error:
                    raise RuntimeError(f"Failed to read NRRD file even after attempting fix: {fix_error}") from e
            else:
                raise

        # SimpleITK returns [D, H, W] already
        original_dtype = str(image.dtype)

        # Get spacing (SimpleITK returns [X, Y, Z], reverse to [Z, X, Y] = [D, H, W])
        spacing_sitk = sitk_image.GetSpacing()
        if len(spacing_sitk) >= 3:
            spacing = [float(spacing_sitk[2]), float(spacing_sitk[0]), float(spacing_sitk[1])]
        elif len(spacing_sitk) == 2:
            spacing = [1.0, float(spacing_sitk[0]), float(spacing_sitk[1])]
        else:
            spacing = [1.0, 1.0, 1.0]

        # Build metadata
        metadata = {
            'spacing': spacing,
            'shape': list(image.shape),
            'dtype': str(image.dtype),
            'original_dtype': original_dtype,
            'origin': list(sitk_image.GetOrigin()),
            'direction': list(sitk_image.GetDirection())
        }

        return {
            'image': image,
            'metadata': metadata,
            'source_format': 'nrrd'
        }

    elif HAS_NRRD:
        # Fallback to pynrrd
        data, header = nrrd.read(str(filepath))
        original_dtype = str(data.dtype)

        # Convert to [D, H, W] format if needed
        # NRRD can have various axis orderings, but commonly uses (X, Y, Z)
        if len(data.shape) == 2:
            # 2D image: [X, Y] -> [1, X, Y]
            image = np.expand_dims(data, axis=0)
        elif len(data.shape) == 3:
            # 3D volume: Assume (X, Y, Z), convert to [D, H, W] = [Z, X, Y]
            image = np.transpose(data, (2, 0, 1))
        else:
            raise ValueError(f"Unsupported NRRD shape: {data.shape}")

        # Get spacing from header
        spacing_nrrd = header.get('space directions', None)
        if spacing_nrrd is not None and len(spacing_nrrd) >= 3:
            # Extract diagonal elements (voxel spacing)
            spacing = [
                float(np.linalg.norm(spacing_nrrd[2])),  # D (Z)
                float(np.linalg.norm(spacing_nrrd[0])),  # H (X)
                float(np.linalg.norm(spacing_nrrd[1]))   # W (Y)
            ]
        else:
            # Try 'spacings' field as fallback
            spacings = header.get('spacings', [1.0, 1.0, 1.0])
            if len(spacings) >= 3:
                spacing = [float(spacings[2]), float(spacings[0]), float(spacings[1])]
            else:
                spacing = [1.0, 1.0, 1.0]

        # Build metadata
        metadata = {
            'spacing': spacing,
            'shape': list(image.shape),
            'dtype': str(image.dtype),
            'original_dtype': original_dtype,
            'nrrd_header': header
        }

        return {
            'image': image,
            'metadata': metadata,
            'source_format': 'nrrd'
        }

    else:
        raise ImportError(
            "NRRD support requires either SimpleITK or pynrrd. "
            "Install with: pip install SimpleITK or pip install pynrrd"
        )


def save_medical_image(
    filepath: Union[str, Path],
    image: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    format: str = 'nifti'
) -> None:
    """
    Save medical image from [D, H, W] format.

    Args:
        filepath: Output path
        image: Image array in [D, H, W] format
        metadata: Optional metadata dict with 'spacing', etc.
        format: Output format ('nifti', 'mha', 'nrrd') - default 'nifti'

    Raises:
        ValueError: If image not in [D, H, W] format
    """
    filepath = Path(filepath)

    if len(image.shape) != 3:
        raise ValueError(f"Image must be in [D, H, W] format, got shape {image.shape}")

    if metadata is None:
        metadata = {}

    suffix = filepath.suffix.lower()
    suffixes = ''.join(filepath.suffixes).lower()

    # Check for NRRD first (before NIfTI) to handle .seg.nrrd correctly
    if format == 'nrrd' or suffix == '.nrrd' or suffixes.endswith('.seg.nrrd'):
        _save_nrrd(filepath, image, metadata)
    elif format == 'nifti' or suffix in ['.nii', '.gz']:
        _save_nifti(filepath, image, metadata)
    elif format == 'mha' or suffix in ['.mha', '.mhd']:
        if not HAS_SITK:
            raise ImportError("SimpleITK required for MHA format. Install with: pip install SimpleITK")
        _save_mha(filepath, image, metadata)
    else:
        # Default to NIfTI
        _save_nifti(filepath, image, metadata)


def _save_nifti(filepath: Path, image: np.ndarray, metadata: Dict[str, Any]) -> None:
    """Save as NIfTI file."""
    # Convert [D, H, W] to NIfTI (X, Y, Z)
    image_nifti = np.transpose(image, (1, 2, 0))

    # Create affine from spacing
    spacing = metadata.get('spacing', [1.0, 1.0, 1.0])
    # NIfTI affine: [X, Y, Z] spacing
    affine = np.diag([spacing[1], spacing[2], spacing[0], 1.0])

    # Create NIfTI
    nii = nib.Nifti1Image(image_nifti, affine)

    # Ensure .nii.gz extension
    if not str(filepath).endswith('.nii.gz'):
        filepath = filepath.with_suffix('.nii.gz')

    nib.save(nii, str(filepath))
    logger.info(f"Saved NIfTI: {filepath}")


def _save_mha(filepath: Path, image: np.ndarray, metadata: Dict[str, Any]) -> None:
    """Save as MHA file using SimpleITK."""
    # Create SimpleITK image from [D, H, W] array
    sitk_image = sitk.GetImageFromArray(image)

    # Set spacing (SimpleITK expects [X, Y, Z], convert from [D, H, W])
    spacing = metadata.get('spacing', [1.0, 1.0, 1.0])
    sitk_image.SetSpacing([spacing[1], spacing[2], spacing[0]])

    # Set origin if available
    if 'origin' in metadata:
        sitk_image.SetOrigin(metadata['origin'])

    # Set direction if available
    if 'direction' in metadata:
        sitk_image.SetDirection(metadata['direction'])

    sitk.WriteImage(sitk_image, str(filepath))
    logger.info(f"Saved MHA: {filepath}")


def _save_nrrd(filepath: Path, image: np.ndarray, metadata: Dict[str, Any]) -> None:
    """
    Save as NRRD file.

    Supports both .nrrd and .seg.nrrd extensions.
    Uses SimpleITK if available, otherwise falls back to pynrrd library.
    """
    # Prefer SimpleITK as it handles spacing/orientation consistently
    if HAS_SITK:
        # Create SimpleITK image from [D, H, W] array
        sitk_image = sitk.GetImageFromArray(image)

        # Set spacing (SimpleITK expects [X, Y, Z], convert from [D, H, W])
        spacing = metadata.get('spacing', [1.0, 1.0, 1.0])
        sitk_image.SetSpacing([spacing[1], spacing[2], spacing[0]])

        # Set origin if available
        if 'origin' in metadata:
            sitk_image.SetOrigin(metadata['origin'])

        # Set direction if available
        if 'direction' in metadata:
            sitk_image.SetDirection(metadata['direction'])

        sitk.WriteImage(sitk_image, str(filepath))
        logger.info(f"Saved NRRD: {filepath}")

    elif HAS_NRRD:
        # Fallback to pynrrd
        # Convert [D, H, W] back to (X, Y, Z) for NRRD
        image_nrrd = np.transpose(image, (1, 2, 0))

        # Build NRRD header
        spacing = metadata.get('spacing', [1.0, 1.0, 1.0])
        header = {
            'type': str(image.dtype),
            'dimension': 3,
            'space': 'left-posterior-superior',
            'sizes': list(image_nrrd.shape),
            'space directions': [
                [spacing[1], 0, 0],  # X spacing (from H)
                [0, spacing[2], 0],  # Y spacing (from W)
                [0, 0, spacing[0]]   # Z spacing (from D)
            ],
            'encoding': 'gzip'
        }

        # Add original NRRD header fields if available
        if 'nrrd_header' in metadata:
            original_header = metadata['nrrd_header']
            # Preserve some fields from original header
            for key in ['space origin', 'kinds', 'endian']:
                if key in original_header:
                    header[key] = original_header[key]

        nrrd.write(str(filepath), image_nrrd, header)
        logger.info(f"Saved NRRD: {filepath}")

    else:
        raise ImportError(
            "NRRD support requires either SimpleITK or pynrrd. "
            "Install with: pip install SimpleITK or pip install pynrrd"
        )


def validate_dhw_format(image: np.ndarray) -> bool:
    """
    Validate that image is in [D, H, W] format.

    Args:
        image: Image array to validate

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be numpy.ndarray")

    if len(image.shape) != 3:
        raise ValueError(f"Image must be 3D [D, H, W], got shape {image.shape}")

    D, H, W = image.shape

    if D < 1:
        raise ValueError(f"Depth (D) must be >= 1, got {D}")

    if H <= 0 or W <= 0:
        raise ValueError(f"Height and Width must be > 0, got H={H}, W={W}")

    if np.any(np.isnan(image)):
        raise ValueError("Image contains NaN values")

    if np.any(np.isinf(image)):
        raise ValueError("Image contains Inf values")

    return True

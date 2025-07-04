"""Image and mask loading utilities."""

import logging
import numpy as np
from pathlib import Path
from typing import Sequence
from skimage import measure, color
from skimage.io import imread

def load_label_mask(mask_path, fix_brightness_artifact=True):
    """Return a *labelled* mask (0 = background) and optional intensity data."""
    seg_path = mask_path.with_name(mask_path.stem.replace("_mask", "_seg") + ".npy")
    
    intensity_data = None
    
    if seg_path.exists():
        try:
            data = np.load(seg_path, allow_pickle=True)
            
            # Handle different numpy save formats
            if hasattr(data, 'item'):
                data = data.item()
            
            if isinstance(data, dict):
                if "masks" in data:
                    labels = data["masks"].astype(np.int32)
                    
                    # FIX BRIGHTNESS ARTIFACT HERE
                    if fix_brightness_artifact and labels.max() > labels.shape[0]:
                        # Likely has the artifact - rebinarize and relabel
                        binary = (labels > 0).astype(np.uint8)
                        labels = measure.label(binary, connectivity=2)
                    
                    # Try to extract intensity information if available
                    if "img" in data:
                        intensity_data = data["img"]
                        if intensity_data.ndim == 3:
                            intensity_data = color.rgb2gray(intensity_data)
                    return labels, intensity_data
                else:
                    logging.warning(f"No 'masks' key found in {seg_path}, available keys: {data.keys()}")
            elif isinstance(data, np.ndarray):
                labels = data.astype(np.int32)
                
                # FIX BRIGHTNESS ARTIFACT HERE TOO
                if fix_brightness_artifact and labels.max() > labels.shape[0]:
                    binary = (labels > 0).astype(np.uint8)
                    labels = measure.label(binary, connectivity=2)
                    
                return labels, None
            else:
                logging.warning(f"Unexpected data type in {seg_path}, falling back to binary mask")
        except Exception as e:
            logging.warning(f"Failed to load {seg_path}: {e}, falling back to binary mask")
    
    # Fallback to binary mask processing
    try:
        bin_mask = imread(mask_path) > 0  # This already fixes the artifact!
        labeled_mask = measure.label(bin_mask, connectivity=2)
        return labeled_mask, None 
    except Exception as e:
        raise ValueError(f"Cannot load mask from {mask_path}: {e}")


def gather_inputs(paths: Sequence[str]) -> list[Path]:
    """Gather all mask files from input paths."""
    # Supported mask extensions
    mask_patterns = ["*_mask.tif", "*_mask.tiff", "*_mask.png", "*_mask.jpg", "*_mask.jpeg",
                     "*_mask.TIF", "*_mask.TIFF", "*_mask.PNG", "*_mask.JPG", "*_mask.JPEG"]
    
    out: list[Path] = []
    for p in paths:
        pth = Path(p)
        if pth.is_dir():
            for pattern in mask_patterns:
                out.extend(pth.rglob(pattern))
        elif pth.is_file() and "_mask." in pth.name:
            # Check if it has a supported extension
            if any(pth.name.endswith(ext) for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg', 
                                                       '.TIF', '.TIFF', '.PNG', '.JPG', '.JPEG']):
                out.append(pth)
    return sorted(set(out))  # Remove duplicates and sort
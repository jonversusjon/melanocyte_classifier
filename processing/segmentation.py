"""Segmentation and overlay utilities."""

import logging
import numpy as np
import cv2
from pathlib import Path
from skimage.io import imread


from ..io.loaders import load_label_mask
from ..io.matchers import find_original_image

def create_overlay_on_original(mask_path, classification_df, image_base_dir=None):
    """Overlay colored outlines on original image based on classification."""
    # Find the original image
    if image_base_dir:
        orig_path = find_original_image(mask_path, Path(image_base_dir))
    else:
        # Try to find it in the same directory structure
        base = mask_path.stem.replace("_mask", "")
        possible_paths = [
            mask_path.parent / f"{base}.png",
            mask_path.parent / f"{base}.tif",
            mask_path.parent / f"{base}.tiff",
            mask_path.parent / f"{base}.jpg",
        ]
        orig_path = None
        for p in possible_paths:
            if p.exists():
                orig_path = p
                break
    
    if not orig_path or not orig_path.exists():
        logging.warning(f"No original image found for {mask_path.name}, creating overlay on black background")
        # Fall back to black background
        labels, _ = load_label_mask(mask_path)
        h, w = labels.shape
        original = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        # Load original image
        original = imread(orig_path)
        if original.ndim == 2:  # Convert grayscale to RGB
            original = np.stack([original] * 3, axis=-1)
        
        # Normalize for display
        if original.dtype != np.uint8:
            if original.max() > 255:
                original = (original / original.max() * 255).astype(np.uint8)
            elif original.max() <= 1.0:
                original = (original * 255).astype(np.uint8)
            else:
                original = original.astype(np.uint8)
    
    # Load mask
    labels, _ = load_label_mask(mask_path)
    
    # Get classifications for this image
    file_data = classification_df[classification_df["mask_file"] == mask_path.name]
    
    # Create classification map
    class_map = dict(zip(file_data['label'].astype(int), file_data['class']))
    
    # Create a copy to draw on
    overlay = original.copy()
    
    # Draw outlines on the copy
    for label in np.unique(labels):
        if label == 0:  # Skip background
            continue
            
        # Create binary mask for this cell
        cell_mask = (labels == label).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Choose color based on classification
        if class_map.get(label) == "live":
            color = (0, 255, 0)  # Green for live cells
        else:
            color = (0, 0, 255)  # Red for dead cells
        
        # Draw contours with thick lines for visibility
        cv2.drawContours(overlay, contours, -1, color, 3)
    
    # Add a legend
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(overlay, "Green = Live", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(overlay, "Red = Dead", (10, 60), font, 1, (0, 0, 255), 2)
    
    return overlay
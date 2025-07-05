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
    
    # Calculate cell size statistics for multi-cell detection
    cell_areas = []
    for label in np.unique(labels):
        if label == 0:  # Skip background
            continue
        cell_mask = (labels == label).astype(np.uint8)
        area = np.sum(cell_mask)
        cell_areas.append(area)
    
    # Calculate threshold for unusually large cells (potential multi-cell areas)
    if cell_areas:
        mean_area = np.mean(cell_areas)
        std_area = np.std(cell_areas)
        large_cell_threshold = mean_area + 2 * std_area  # Cells 2 std devs above mean
    else:
        large_cell_threshold = float('inf')
    
    # Draw outlines on the copy
    for label in np.unique(labels):
        if label == 0:  # Skip background
            continue
            
        # Create binary mask for this cell
        cell_mask = (labels == label).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate cell area and shape metrics
        cell_area = np.sum(cell_mask)
        
        # Detect potential multi-cell areas
        is_multi_cell = False
        if contours:
            contour = contours[0]  # Take the largest contour
            # Check if area is unusually large
            if cell_area > large_cell_threshold:
                is_multi_cell = True
            # Check for irregular shape (low solidity)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = cv2.contourArea(contour) / hull_area
                if solidity < 0.7:  # Low solidity indicates irregular shape
                    is_multi_cell = True
        
        # Choose color based on classification and multi-cell detection
        if is_multi_cell:
            # Create dilated contours for magenta outline (outer ring)
            # Dilate the cell mask to create an outer boundary
            kernel = np.ones((9, 9), np.uint8)
            dilated_mask = cv2.dilate(cell_mask, kernel, iterations=1)
            
            # Find contours of dilated mask
            dilated_contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw magenta outline on dilated contours (outer ring)
            cv2.drawContours(overlay, dilated_contours, -1, (255, 0, 255), 3)  # Magenta outer ring
            
        # Draw the main classification outline (inner outline)
        if class_map.get(label) == "live":
            color = (0, 255, 0)  # Green for live cells (BGR format)
        else:
            color = (255, 0, 0)  # Red for dead cells (BGR format - fixed from blue)
        
        # Draw contours with thick lines for visibility
        cv2.drawContours(overlay, contours, -1, color, 3)
    
    # Add a legend
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(overlay, "Green = Live", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(overlay, "Red = Dead", (10, 60), font, 1, (255, 0, 0), 2)
    cv2.putText(overlay, "Magenta = Multi-cell", (10, 90), font, 1, (255, 0, 255), 2)
    
    return overlay
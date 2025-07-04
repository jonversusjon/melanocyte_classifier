"""Overlay generation utilities."""

import logging
import numpy as np
import cv2
from pathlib import Path
from skimage.io import imread, imsave
from ..io.loaders import load_label_mask

def outline_overlay(label_img: np.ndarray, flags: dict[int, str]) -> np.ndarray:
    """Create overlay image with colored outlines based on classification."""
    h, w = label_img.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Find unique labels (excluding background)
    unique_labels = np.unique(label_img)
    unique_labels = unique_labels[unique_labels > 0]
    
    for label in unique_labels:
        # Create binary mask for this label
        binary_mask = (label_img == label).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Determine color based on classification
        color_bgr = (0, 255, 0) if flags.get(label) == "live" else (255, 0, 0)  # Green for live, Red for dead

        # Draw contours
        cv2.drawContours(out, contours, -1, color_bgr, 1)
    
    return out


def generate_example_overlays(mask_paths: list[Path], full_df, 
                            n_examples: int = 5, output_dir: Path = Path("examples")):
    """Generate example overlays on original images."""
    output_dir.mkdir(exist_ok=True)
    
    # Select random examples or first n
    example_paths = mask_paths[:n_examples]
    
    for mask_path in example_paths:
        # Find original image
        base = mask_path.stem.replace("_mask", "")
        possible_orig = [
            mask_path.parent / f"{base}.tif",
            mask_path.parent / f"{base}.tiff",
        ]
        
        orig_path = None
        for p in possible_orig:
            if p.exists():
                orig_path = p
                break
                
        if orig_path:
            from ..processing.segmentation import create_overlay_on_original
            overlay = create_overlay_on_original(mask_path, full_df)
            output_path = output_dir / f"{base}_overlay_on_original.tif"
            imsave(output_path, overlay, check_contrast=False)
            logging.info(f"Saved example overlay: {output_path}")
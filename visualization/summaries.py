"""Summary visualization utilities."""

import numpy as np
import cv2
from skimage.io import imread, imsave

def create_summary_grid(overlay_paths, output_path, max_images=12):
    """Create a grid of overlay images for quick review."""
    # Select up to max_images
    selected_paths = overlay_paths[:max_images]
    
    if not selected_paths:
        return
    
    # Load images
    images = []
    for path in selected_paths:
        img = imread(path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        images.append(img)
    
    # Resize all images to same size (use smallest dimensions)
    min_h = min(img.shape[0] for img in images)
    min_w = min(img.shape[1] for img in images)
    
    resized = []
    for img in images:
        if img.shape[:2] != (min_h, min_w):
            resized_img = cv2.resize(img, (min_w, min_h))
            resized.append(resized_img)
        else:
            resized.append(img)
    
    # Create grid
    n_images = len(resized)
    grid_size = int(np.ceil(np.sqrt(n_images)))
    
    # Create blank grid
    grid = np.ones((grid_size * min_h, grid_size * min_w, 3), dtype=np.uint8) * 255
    
    # Place images
    for idx, img in enumerate(resized):
        row = idx // grid_size
        col = idx % grid_size
        grid[row*min_h:(row+1)*min_h, col*min_w:(col+1)*min_w] = img
    
    # Save grid
    imsave(output_path, grid, check_contrast=False)
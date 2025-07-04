"""Feature extraction for melanocyte classification."""

import numpy as np
import pandas as pd
from skimage import measure

def region_features(label_img: np.ndarray, intensity_img: np.ndarray) -> pd.DataFrame:
    """Extract region properties from labeled image."""
    props = measure.regionprops(label_img, intensity_image=intensity_img)
    rows: list[dict[str, float | int]] = []
    
    for p in props:
        if p.label == 0:
            continue
        
        area = p.area
        perim = p.perimeter or 1.0  # Avoid division by zero
        circularity = 4 * np.pi * area / (perim ** 2)
        
        # Get median intensity for this region
        region_mask = label_img == p.label
        median_int = float(np.median(intensity_img[region_mask]))
        
        rows.append({
            "label": p.label,
            "area": area,
            "perimeter": perim,
            "circularity": circularity,
            "median_int": median_int,
        })
    
    return pd.DataFrame(rows)
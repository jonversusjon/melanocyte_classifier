"""Cell analysis utilities."""

import numpy as np
from skimage import measure

def detect_touching_cells(labels, area_stats=None):
    """Identify cells that might be touching based on area analysis."""
    props = measure.regionprops(labels)
    
    # Calculate area statistics
    areas = [p.area for p in props if p.label > 0]
    if not areas:
        return [], {}
        
    median_area = np.median(areas)
    std_area = np.std(areas)
    
    # Use provided stats or calculate
    if area_stats is not None:
        median_area = area_stats.get('median_area', median_area)
        std_area = area_stats.get('std_area', std_area)
    
    # Cells with unusually large area might be touching
    potentially_touching = []
    
    for prop in props:
        if prop.label == 0:
            continue
            
        # Multiple detection criteria
        area_ratio = prop.area / median_area
        if (area_ratio > 1.8 or  # Nearly 2x median suggests 2 cells
            prop.area > median_area + 2.5 * std_area or  # Statistical outlier
            (prop.area > median_area * 1.5 and prop.solidity < 0.9)):  # Large and non-convex
            
            potentially_touching.append({
                'label': prop.label,
                'area': prop.area,
                'eccentricity': prop.eccentricity,
                'solidity': prop.solidity,
                'centroid': prop.centroid,
                'area_ratio': area_ratio,
                'estimated_cells': round(area_ratio)
            })
    
    stats = {
        'median_area': median_area, 
        'std_area': std_area,
        'total_cells': len(props),
        'touching_cells': len(potentially_touching)
    }
    
    return potentially_touching, stats
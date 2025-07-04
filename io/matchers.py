"""Image-mask matching utilities."""

import logging
from pathlib import Path
from .loaders import gather_inputs

def find_original_image(mask_path: Path, image_base_dir: Path) -> Path | None:
    """Find the original image corresponding to a mask file."""
    mask_name = mask_path.stem  # Remove extension (.tif, .png, etc.)
    
    # Remove _mask suffix
    base_name = mask_name.replace("_mask", "")
    
    # Supported image extensions
    image_extensions = ['.png', '.tif', '.tiff', '.jpg', '.jpeg', '.PNG', '.TIF', '.TIFF', '.JPG', '.JPEG']
    
    # Try different strategies to find the original
    strategies = []
    
    # Strategy 1: Exact match with different extensions
    for ext in image_extensions:
        strategies.append(base_name + ext)
    
    # Strategy 2: Remove GUID (everything after the last underscore that looks like a GUID)
    if len(base_name.rsplit("_", 1)[-1]) >= 32:
        guid_removed = base_name.rsplit("_", 1)[0]
        for ext in image_extensions:
            strategies.append(guid_removed + ext)
    
    # Strategy 3: Remove everything after _w (channel info)
    if "_w" in base_name:
        channel_removed = base_name.split("_w")[0]
        for ext in image_extensions:
            strategies.append(channel_removed + ext)
    
    # Search in all subdirectories of the image base directory
    for strategy in strategies:
        matches = list(image_base_dir.rglob(strategy))
        if matches:
            if len(matches) > 1:
                logging.warning(f"Multiple matches found for {mask_path.name}, using first: {matches[0]}")
            logging.debug(f"Matched {mask_path.name} -> {matches[0].name}")
            return matches[0]
    
    # If no exact match, try more flexible partial matching
    # Look for files that contain key parts of the mask name
    experimental_parts = base_name.split("_")
    
    # Try to identify well position (e.g., A02, B03, etc.)
    well_pattern = None
    for part in experimental_parts:
        if len(part) == 3 and part[0].isalpha() and part[1:].isdigit():
            well_pattern = part
            break
    
    if well_pattern:
        # Search for any image file containing this well pattern
        for pattern in ["*.png", "*.tif*", "*.jpg", "*.jpeg", "*.PNG", "*.TIF*", "*.JPG", "*.JPEG"]:
            for img_path in image_base_dir.rglob(pattern):
                if well_pattern in img_path.name:
                    # Additional check: ensure some experimental context matches
                    if any(part in str(img_path) for part in experimental_parts[:3]):
                        logging.debug(f"Fuzzy matched {mask_path.name} -> {img_path.name}")
                        return img_path
    
    return None


def test_image_matching(mask_dir: str, image_dir: str, n_samples: int = 10):
    """Test image matching without running full analysis."""
    image_path = Path(image_dir)
    
    # Find some mask files
    masks = gather_inputs([mask_dir])[:n_samples]
    
    print(f"\nTesting image matching for {len(masks)} masks...")
    print("=" * 80)
    
    matched = 0
    for mask in masks:
        orig = find_original_image(mask, image_path)
        if orig:
            print(f"✓ {mask.name}")
            print(f"  → {orig.name}")
            matched += 1
        else:
            print(f"✗ {mask.name}")
            print("  → NO MATCH FOUND")
        print()
    
    print(f"Summary: {matched}/{len(masks)} masks matched to original images")
    return matched == len(masks)
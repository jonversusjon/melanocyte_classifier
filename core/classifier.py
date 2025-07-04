"""Main classification logic for melanocytes."""

import logging
import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
from tqdm.auto import tqdm
from skimage import filters, color
from skimage.io import imread

from ..io.loaders import load_label_mask
from ..io.matchers import find_original_image
from .features import region_features

class MelanocyteClassifier:
    """Main classifier for melanocytes (live/dead classification)."""
    
    def __init__(self, workers: int = 1):
        """Initialize classifier.
        
        Args:
            workers: Number of parallel workers for processing
        """
        self.workers = workers
        self.circularity_threshold = None
        self.intensity_threshold = None
        
    def process_single_mask(self, mask_path: Path, orig_dir: Optional[str] = None, 
                           image_base_dir: Optional[Path] = None) -> Tuple[Path, pd.DataFrame]:
        """Process a single mask file."""
        base = mask_path.stem.replace("_mask", "")
        
        # Load labeled mask and try to get intensity from segmentation file
        labels, seg_intensity = load_label_mask(mask_path)
        
        intensity = None
        orig_name = f"{base}.tif"  # Default name for file column
        
        # First priority: use intensity from segmentation file if available
        if seg_intensity is not None:
            intensity = seg_intensity
            logging.debug(f"Using intensity from segmentation file for {mask_path.name}")
        else:
            # Second priority: try to find original image
            orig_path = None
            
            # If image_base_dir is provided, use the new finding logic
            if image_base_dir:
                orig_path = find_original_image(mask_path, Path(image_base_dir))
                if orig_path:
                    logging.debug(f"Found original image: {orig_path}")
            else:
                # Fall back to the old logic
                possible_originals = [
                    mask_path.parent / f"{base}.tif",
                    mask_path.parent / f"{base}.tiff", 
                    mask_path.parent.parent / f"{base}.tif",
                    mask_path.parent.parent / f"{base}.tiff",
                    mask_path.parent / f"{base.rsplit('_', 1)[0]}.tif" if '_' in base else None,
                ]
                
                if orig_dir:
                    orig_path_base = Path(orig_dir)
                    possible_originals.extend([
                        orig_path_base / f"{base}.tif",
                        orig_path_base / f"{base}.tiff",
                        orig_path_base / f"{base.rsplit('_', 1)[0]}.tif" if '_' in base else None,
                    ])
                
                possible_originals = [p for p in possible_originals if p is not None]
                
                for p in possible_originals:
                    if p.exists():
                        orig_path = p
                        break
            
            if orig_path and orig_path.exists():
                intensity = imread(orig_path)
                if intensity.ndim == 3:
                    intensity = color.rgb2gray(intensity)
                orig_name = orig_path.name
            
        # If still no intensity found, use mask as fallback
        if intensity is None:
            logging.warning(f"No original image found for {mask_path.name}, using mask for intensity")
            mask_img = imread(mask_path)
            if mask_img.ndim == 3:
                intensity = color.rgb2gray(mask_img)
            else:
                intensity = mask_img.astype(float)
            if intensity.max() > 0:
                intensity = intensity / intensity.max()
        
        # Extract features
        df = region_features(labels, intensity)
        df["file"] = orig_name
        df["mask_file"] = mask_path.name
        
        return mask_path, df
    
    def fit(self, masks: List[Path], orig_dir: Optional[str] = None, 
            image_base_dir: Optional[Path] = None) -> 'MelanocyteClassifier':
        """Fit classifier thresholds on mask data.
        
        Args:
            masks: List of mask file paths
            orig_dir: Directory containing original images (optional)
            image_base_dir: Base directory for finding original images (optional)
            
        Returns:
            self
        """
        logging.info(f"Processing {len(masks)} masks to determine thresholds...")
        
        # Process all masks
        dfs: list[pd.DataFrame] = []
        circ_all: list[float] = []
        int_all: list[float] = []

        iterable = [(p, orig_dir, image_base_dir) for p in masks]
        
        # Handle multiprocessing properly
        if self.workers > 1:
            with mp.Pool(self.workers) as pool:
                results = list(tqdm(
                    pool.imap_unordered(self._process_one_wrapper, iterable), 
                    total=len(masks), 
                    desc="Processing masks"
                ))
        else:
            results = list(tqdm(
                map(self._process_one_wrapper, iterable), 
                total=len(masks), 
                desc="Processing masks"
            ))
        
        # Collect results
        for mask_path, df in results:
            dfs.append(df)
            circ_all.extend(df["circularity"].tolist())
            int_all.extend(df["median_int"].tolist())

        # Calculate thresholds using Otsu's method
        self.circularity_threshold = float(filters.threshold_otsu(np.asarray(circ_all)))
        self.intensity_threshold = float(filters.threshold_otsu(np.asarray(int_all)))

        logging.info(f"Thresholds: circularity > {self.circularity_threshold:.3f}, "
                    f"median_int < {self.intensity_threshold:.1f}")
        
        return self
    
    def _process_one_wrapper(self, args: tuple) -> Tuple[Path, pd.DataFrame]:
        """Wrapper for multiprocessing."""
        mask_path, orig_dir, image_base_dir = args
        return self.process_single_mask(mask_path, orig_dir, image_base_dir)
    
    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify cells in a dataframe.
        
        Args:
            df: DataFrame with circularity and median_int columns
            
        Returns:
            DataFrame with added 'class' column
        """
        if self.circularity_threshold is None or self.intensity_threshold is None:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        df = df.copy()
        df["class"] = np.where(
            (df["circularity"] > self.circularity_threshold) & 
            (df["median_int"] < self.intensity_threshold), 
            "dead", 
            "live"
        )
        return df
    
    def fit_predict(self, masks: List[Path], orig_dir: Optional[str] = None, 
                   image_base_dir: Optional[Path] = None) -> pd.DataFrame:
        """Fit classifier and predict on the same data.
        
        Args:
            masks: List of mask file paths
            orig_dir: Directory containing original images (optional)
            image_base_dir: Base directory for finding original images (optional)
            
        Returns:
            DataFrame with classifications
        """
        # Process all masks
        dfs: list[pd.DataFrame] = []
        circ_all: list[float] = []
        int_all: list[float] = []

        iterable = [(p, orig_dir, image_base_dir) for p in masks]
        
        # Handle multiprocessing properly
        if self.workers > 1:
            with mp.Pool(self.workers) as pool:
                results = list(tqdm(
                    pool.imap_unordered(self._process_one_wrapper, iterable), 
                    total=len(masks), 
                    desc="Processing masks"
                ))
        else:
            results = list(tqdm(
                map(self._process_one_wrapper, iterable), 
                total=len(masks), 
                desc="Processing masks"
            ))
        
        # Collect results
        for mask_path, df in results:
            dfs.append(df)
            circ_all.extend(df["circularity"].tolist())
            int_all.extend(df["median_int"].tolist())

        # Calculate thresholds using Otsu's method
        self.circularity_threshold = float(filters.threshold_otsu(np.asarray(circ_all)))
        self.intensity_threshold = float(filters.threshold_otsu(np.asarray(int_all)))

        logging.info(f"Thresholds: circularity > {self.circularity_threshold:.3f}, "
                    f"median_int < {self.intensity_threshold:.1f}")

        # Combine all dataframes and classify
        full_df = pd.concat(dfs, ignore_index=True)
        return self.classify(full_df)
"""Command-line interface for melanocyte classification."""

import argparse
import logging
import pandas as pd
from pathlib import Path
from typing import Sequence
from tqdm.auto import tqdm

from .core.classifier import MelanocyteClassifier
from .io.loaders import gather_inputs
from .io.matchers import test_image_matching
from .processing.analysis import detect_touching_cells
from .processing.segmentation import create_overlay_on_original
from .visualization.overlays import outline_overlay
from .visualization.summaries import create_summary_grid
from .utils.helpers import setup_logging

def main(argv: Sequence[str] | None = None):
    """Main CLI entry point."""
    ap = argparse.ArgumentParser(description="Classify melanocytes (live/dead)")
    ap.add_argument("inputs", nargs="+", help="mask files or directories")
    ap.add_argument("--overlay", action="store_true", help="save overlay images (optional - for visualization only)")
    ap.add_argument("--workers", type=int, default=1, help="CPU workers (>=1)")
    ap.add_argument("--debug", action="store_true", help="force single‑process & verbose")
    ap.add_argument("--orig-dir", type=str, help="directory containing original images (if different from mask directory)")
    ap.add_argument("--image-base-dir", type=str, help="base directory containing all original images")
    ap.add_argument("--sample", type=int, help="randomly sample N files for testing")
    ap.add_argument("--seed", type=int, default=42, help="random seed for sampling (default: 42)")
    ap.add_argument("--output-dir", type=str, help="directory for all output files (default: current directory)")
    args = ap.parse_args(argv)

    # Set up logging
    setup_logging(args.debug)

    # Create output directory if specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directory: {output_dir}")
    else:
        output_dir = Path.cwd()
        logging.info(f"Output directory: {output_dir} (current directory)")

    if args.debug:
        args.workers = 1

    masks = gather_inputs(args.inputs)
    if not masks:
        raise SystemExit("No *_mask.tif files found in given inputs.")

    # Add random sampling if requested
    if args.sample:
        import random
        random.seed(args.seed)  # For reproducibility
        
        original_count = len(masks)
        sample_size = min(args.sample, len(masks))
        masks = random.sample(masks, sample_size)
        
        logging.info(f"Found {original_count} mask files total")
        logging.info(f"Randomly selected {sample_size} files for testing (seed={args.seed})")
        logging.info("Selected files:")
        for i, mask in enumerate(masks, 1):
            logging.info(f"  {i}. {mask.name}")
    else:
        logging.info(f"Found {len(masks)} mask files to process")

    # Create classifier and run analysis
    classifier = MelanocyteClassifier(workers=args.workers)
    full_df = classifier.fit_predict(
        masks, 
        orig_dir=args.orig_dir, 
        image_base_dir=args.image_base_dir
    )

    # Detect touching cells
    all_touching = []
    for mask_path in masks:
        from .io.loaders import load_label_mask
        labels, _ = load_label_mask(mask_path)
        touching, stats = detect_touching_cells(labels)
        if touching:
            for cell in touching:
                cell['file'] = mask_path.name
            all_touching.extend(touching)

    if all_touching:
        touching_df = pd.DataFrame(all_touching)
        touching_df.to_csv(output_dir / "potentially_touching_cells.csv", index=False)
        logging.info(f"Found {len(all_touching)} potentially touching cells across all images")

    # Generate overlays if requested
    if args.overlay:
        logging.info("Generating overlay images on original images...")
        overlay_dir = output_dir / "overlays_on_originals"
        overlay_dir.mkdir(exist_ok=True)
        
        # Also create a directory for outline-only overlays
        outline_dir = output_dir / "outlines_only"
        outline_dir.mkdir(exist_ok=True)
        
        for mask_path in tqdm(masks, desc="Creating overlays"):
            base = mask_path.stem.replace("_mask", "")
            
            # Check if we have classification data for this file
            file_data = full_df[full_df["mask_file"] == mask_path.name]
            if file_data.empty:
                continue
            
            # Generate overlay on original image
            try:
                overlay_on_original = create_overlay_on_original(
                    mask_path, 
                    full_df, 
                    image_base_dir=args.image_base_dir
                )
                
                # Save overlay on original
                from skimage.io import imsave
                overlay_path = overlay_dir / f"{base}_overlay_on_original.png"
                imsave(overlay_path, overlay_on_original, check_contrast=False)
                
            except Exception as e:
                logging.error(f"Failed to create overlay for {mask_path.name}: {e}")
            
            # Create outline-only version
            flag_map = dict(zip(file_data['label'].astype(int), file_data['class'].astype(str)))
            from .io.loaders import load_label_mask
            labels, _ = load_label_mask(mask_path)
            outline_img = outline_overlay(labels, flag_map)
            outline_path = outline_dir / f"{base}_outlines_only.png"
            imsave(outline_path, outline_img, check_contrast=False)
    
    # Create summary grid
    if args.overlay:
        overlay_files = list((output_dir / "overlays_on_originals").glob("*.png"))
        if overlay_files:
            logging.info("Creating summary grid...")
            create_summary_grid(
                overlay_files, 
                output_dir / "summary_grid.png",
                max_images=min(12, len(overlay_files))
            )

    # Save per-image CSV files
    for mask_path in masks:
        base = mask_path.stem.replace("_mask", "")
        per_file_data = full_df[full_df["mask_file"] == mask_path.name]
        if not per_file_data.empty:
            csv_path = output_dir / f"{base}_classification.csv"
            per_file_data.to_csv(csv_path, index=False)

    # Save summary
    summary_df = full_df.copy()
    summary_df.to_csv(output_dir / "classification_summary.csv", index=False)
    
    # Print detailed summary statistics
    live_count = (summary_df["class"] == "live").sum()
    dead_count = (summary_df["class"] == "dead").sum()
    total_cells = len(summary_df)
    
    # Per-file statistics
    file_stats = summary_df.groupby('mask_file')['class'].value_counts().unstack(fill_value=0)
    if 'live' not in file_stats.columns:
        file_stats['live'] = 0
    if 'dead' not in file_stats.columns:
        file_stats['dead'] = 0
    
    file_stats['total'] = file_stats['live'] + file_stats['dead']
    file_stats['live_pct'] = (file_stats['live'] / file_stats['total'] * 100).round(1)
    file_stats['dead_pct'] = (file_stats['dead'] / file_stats['total'] * 100).round(1)
    
    # Save per-file summary
    file_stats.to_csv(output_dir / "per_file_summary.csv")
    
    print("\n✔ CLASSIFICATION COMPLETE")
    print("=" * 50)
    if args.sample:
        print(f"TEST RUN - Sampled {len(masks)} files from {len(gather_inputs(args.inputs))} total")
    print(f"Total files processed: {len(masks)}")
    print(f"Total cells analyzed: {total_cells:,}")
    print(f"Live cells: {live_count:,} ({live_count/total_cells*100:.1f}%)")
    print(f"Dead cells: {dead_count:,} ({dead_count/total_cells*100:.1f}%)")
    print("Classification thresholds:")
    print(f"  - Circularity > {classifier.circularity_threshold:.3f}")
    print(f"  - Median intensity < {classifier.intensity_threshold:.1f}")
    print("\nOutput files:")
    print("  - classification_summary.csv (all cells)")
    print("  - per_file_summary.csv (counts per image)")
    print("  - Individual CSV files for each image")
    if args.overlay:
        print("  - Overlay images in overlays_on_originals/")
    print("=" * 50)


def cli_main():
    """Entry point for console script."""
    import sys
    
    # Handle test mode
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        if len(sys.argv) < 4:
            print("Usage: melanocyte-classifier test <mask_dir> <image_dir> [n_samples]")
            print("Example: melanocyte-classifier test '/path/to/masks' '/path/to/images' 10")
            sys.exit(1)
        
        mask_dir = sys.argv[2]
        image_dir = sys.argv[3]
        n_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        
        print("Testing image matching...")
        print(f"Mask directory: {mask_dir}")
        print(f"Image directory: {image_dir}")
        print(f"Number of samples: {n_samples}")
        
        test_image_matching(mask_dir, image_dir, n_samples)
        sys.exit(0)
    else:
        main()
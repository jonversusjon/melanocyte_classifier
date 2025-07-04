# Melanocyte Classifier

A scientific tool for classifying melanocytes as live or dead based on microscopy images. This package processes Cellpose-SAM segmentations of melanocytes and classifies them based on morphological features like circularity and intensity.

## Features

- **Automated Classification**: Classify melanocytes as live (amoeboid, bright) or dead (circular, dark)
- **Batch Processing**: Process multiple images with multiprocessing support
- **Flexible Input**: Support for various image formats and directory structures
- **Visualization**: Generate overlay images showing classifications
- **Scientific Reproducibility**: Detailed logging and parameter tracking
- **Easy Installation**: pip-installable package with clear dependencies

## Installation

### Option 1: Install from PyPI (when published)
```bash
pip install melanocyte-classifier
```

### Option 2: Install from source
```bash
git clone https://github.com/yourorg/melanocyte-classifier.git
cd melanocyte-classifier
pip install -e .
```

### Option 3: Install dependencies manually
```bash
conda install -c conda-forge scikit-image opencv pandas tqdm tifffile
# or
pip install scikit-image opencv-python pandas tqdm tifffile
```

## Quick Start

### Command Line Usage

```bash
# Basic classification
melanocyte-classifier /path/to/masks --workers 8

# With overlay generation
melanocyte-classifier /path/to/masks --overlay --workers 8

# Specify original images directory
melanocyte-classifier /path/to/masks --image-base-dir /path/to/original/images --overlay

# Test a subset of images
melanocyte-classifier /path/to/masks --sample 10 --seed 42
```

### Python API Usage

```python
from melanocyte_classifier import MelanocyteClassifier
from melanocyte_classifier.io import gather_inputs

# Create classifier
classifier = MelanocyteClassifier(workers=4)

# Find mask files
masks = gather_inputs(["/path/to/masks"])

# Classify cells
results = classifier.fit_predict(masks)
print(results.head())

# Save results
results.to_csv("classification_results.csv", index=False)
```

## Scientific Background

This tool implements a morphological classification approach for melanocytes:

- **Live cells**: Typically amoeboid (low circularity) and bright (high intensity)
- **Dead cells**: Typically circular (high circularity) and dark (low intensity)

The classification uses Otsu's method to automatically determine optimal thresholds for:
- Circularity: 4πA/P² (where A=area, P=perimeter)  
- Median intensity: within each cell region

## File Structure

The package expects the following input structure:

```
data/
├── experiment1/
│   ├── image1_mask.tif          # Segmentation masks
│   ├── image1_seg.npy           # Optional: Cellpose segmentation data
│   └── image1.tif               # Original images (for overlay)
└── experiment2/
    ├── image2_mask.tif
    └── image2.tif
```

## Output Files

- `classification_summary.csv`: All cells with features and classifications
- `per_file_summary.csv`: Summary statistics per image
- `potentially_touching_cells.csv`: Cells that may be touching/overlapping
- `overlays_on_originals/`: Overlay images showing classifications
- `outlines_only/`: Outline-only classification images
- `summary_grid.png`: Grid overview of overlay images

## Advanced Usage

### Testing Image Matching

Before running full analysis, test if mask files can be matched to original images:

```bash
melanocyte-classifier test /path/to/masks /path/to/images 10
```

### Programmatic Usage

```python
from melanocyte_classifier.core import MelanocyteClassifier
from melanocyte_classifier.io import find_original_image, load_label_mask
from melanocyte_classifier.processing import detect_touching_cells
from melanocyte_classifier.visualization import create_summary_grid

# Load and analyze a single image
mask_path = Path("image_mask.tif")
labels, intensity = load_label_mask(mask_path)

# Detect potentially touching cells
touching_cells, stats = detect_touching_cells(labels)
print(f"Found {len(touching_cells)} potentially touching cells")

# Create classifier and fit on data
classifier = MelanocyteClassifier(workers=4)
classifier.fit(mask_files)

# Manual classification with custom thresholds
classifier.circularity_threshold = 0.6
classifier.intensity_threshold = 100
results = classifier.classify(feature_dataframe)
```

## Command Line Options

- `inputs`: Mask files or directories (required)
- `--overlay`: Generate overlay visualization images
- `--workers N`: Number of parallel workers (default: 1)
- `--debug`: Enable debug logging and single-process mode
- `--orig-dir`: Directory containing original images
- `--image-base-dir`: Base directory for recursive image search
- `--sample N`: Process only N randomly selected files
- `--seed N`: Random seed for sampling (default: 42)
- `--output-dir`: Output directory (default: current directory)

## Backwards Compatibility

Existing scripts using the original `classify_melanocytes.py` will continue to work:

```bash
python classify_melanocytes.py /path/to/masks --overlay --workers 8
```

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

### Code Formatting

```bash
black melanocyte_classifier/
flake8 melanocyte_classifier/
```

### Building Documentation

```bash
pip install -e ".[docs]"
cd docs/
make html
```

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{melanocyte_classifier,
  title={Melanocyte Classifier: Automated Classification of Live and Dead Melanocytes},
  author={Scientific Computing Team},
  year={2024},
  url={https://github.com/yourorg/melanocyte-classifier}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **Issues**: Report bugs at https://github.com/yourorg/melanocyte-classifier/issues
- **Discussions**: Ask questions at https://github.com/yourorg/melanocyte-classifier/discussions
- **Email**: support@example.com

## Changelog

### v1.0.0 (2024)
- Initial release with modular package structure
- Improved error handling and logging
- Added programmatic API
- Enhanced documentation
- Backwards compatibility maintained
- Added pip installation support
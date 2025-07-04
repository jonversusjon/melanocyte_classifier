"""
Melanocyte Classifier - A scientific tool for classifying melanocytes as live or dead.

This package provides tools for processing microscopy images of melanocytes and
classifying them based on morphological features like circularity and intensity.
"""

__version__ = "1.0.0"
__author__ = "Scientific Computing Team"
__email__ = "support@example.com"

from .core.classifier import MelanocyteClassifier
from .core.features import region_features
from .io.loaders import load_label_mask
from .io.matchers import find_original_image

__all__ = [
    "MelanocyteClassifier",
    "region_features", 
    "load_label_mask",
    "find_original_image"
]
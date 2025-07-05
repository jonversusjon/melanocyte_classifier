#!/usr/bin/env python3
"""
Backwards compatibility script for classify_melanocytes.py.

This script maintains compatibility with the original command-line interface
while using the new modular package structure.
"""

import sys
import os

# Add the package directory to Python path if running as a script
if __name__ == "__main__":
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add parent directory to path so we can import the package
    parent_dir = os.path.dirname(script_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

# Import the main CLI function
from .cli import main

if __name__ == "__main__":
    main()
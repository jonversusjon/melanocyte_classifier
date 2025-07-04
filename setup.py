"""Setup script for melanocyte_classifier package."""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read version from __init__.py
def read_version():
    version_path = os.path.join(os.path.dirname(__file__), 'melanocyte_classifier', '__init__.py')
    with open(version_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"\'')
    return "0.0.0"

setup(
    name="melanocyte-classifier",
    version=read_version(),
    description="A scientific tool for classifying melanocytes as live or dead based on microscopy images",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Scientific Computing Team",
    author_email="support@example.com",
    url="https://github.com/yourorg/melanocyte-classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "scikit-image>=0.19.0",
        "opencv-python>=4.5.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "tqdm>=4.60.0",
        "tifffile>=2021.7.2",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "sphinx-autodoc-typehints>=1.12",
        ],
    },
    entry_points={
        "console_scripts": [
            "melanocyte-classifier=melanocyte_classifier.cli:cli_main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="melanocyte classification microscopy image-processing biology",
    project_urls={
        "Bug Reports": "https://github.com/yourorg/melanocyte-classifier/issues",
        "Source": "https://github.com/yourorg/melanocyte-classifier",
        "Documentation": "https://melanocyte-classifier.readthedocs.io/",
    },
)
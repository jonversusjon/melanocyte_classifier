"""Helper utilities."""

import logging

def setup_logging(debug: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        format="[%(levelname)s] %(message)s", 
        level=level
    )
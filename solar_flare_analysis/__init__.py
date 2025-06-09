"""Solar Flare Analysis Package

A comprehensive system for analyzing solar flare data using ML models,
including data processing, simulation, and visualization.
"""

__version__ = "1.0.0"

# Expose top-level modules
from .src import data_processing, ml_models, visualization

__all__ = [
    "data_processing",
    "ml_models", 
    "visualization"
]

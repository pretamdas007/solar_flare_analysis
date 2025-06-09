#!/usr/bin/env python3
"""
Script to create clean __init__.py files without encoding issues
"""

import os

def create_clean_init_file(filepath, content):
    """Create a clean __init__.py file with proper UTF-8 encoding"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Write with explicit UTF-8 encoding
        with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)
        
        print(f"Created clean file: {filepath}")
    except Exception as e:
        print(f"Error creating {filepath}: {e}")

# Main project __init__.py
main_init_content = '''"""Solar Flare Analysis Package

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
'''

# Data processing __init__.py
data_processing_init_content = '''"""Data processing package"""

from .preprocessor import DataPreprocessor

__all__ = ['DataPreprocessor']
'''

# ML models __init__.py
ml_models_init_content = '''"""ML models package"""

from .enhanced_model import EnhancedSolarFlarePredictor

__all__ = ['EnhancedSolarFlarePredictor']
'''

# Visualization __init__.py
visualization_init_content = '''"""Visualization package"""

from .plotter import SolarFlareVisualizer

__all__ = ['SolarFlareVisualizer']
'''

# Other submodule __init__.py files
simple_init_content = '''"""Package module"""
'''

if __name__ == "__main__":
    base_path = r"c:\Users\srabani\Desktop\goesflareenv\solar_flare_analysis"
    
    # Create main __init__.py
    create_clean_init_file(
        os.path.join(base_path, "__init__.py"),
        main_init_content
    )
    
    # Create src/__init__.py (already clean, but ensure it's good)
    src_init_content = '''"""
Solar flare analysis source package.

This package contains the core modules for solar flare data analysis,
including data processing, ML models, and visualization tools.
"""

__version__ = "1.0.0"

# Import main submodules
from . import data_processing
from . import ml_models
from . import visualization

__all__ = [
    'data_processing',
    'ml_models',
    'visualization'
]
'''
    
    create_clean_init_file(
        os.path.join(base_path, "src", "__init__.py"),
        src_init_content
    )
    
    # Create submodule __init__.py files
    submodules = [
        ("src", "data_processing", data_processing_init_content),
        ("src", "ml_models", ml_models_init_content),
        ("src", "visualization", visualization_init_content),
        ("src", "analysis", simple_init_content),
        ("src", "evaluation", simple_init_content),
        ("src", "flare_detection", simple_init_content),
        ("src", "validation", simple_init_content),
    ]
    
    for submodule_path_parts in submodules:
        if len(submodule_path_parts) == 3:
            path_parts, content = submodule_path_parts[:-1], submodule_path_parts[-1]
        else:
            path_parts, content = submodule_path_parts, simple_init_content
            
        filepath = os.path.join(base_path, *path_parts, "__init__.py")
        create_clean_init_file(filepath, content)

    print("\nAll clean __init__.py files created successfully!")

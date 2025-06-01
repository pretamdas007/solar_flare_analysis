#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify all required packages are installed correctly.
"""

import sys
import importlib

# List of required packages
required_packages = [
    'numpy',
    'pandas', 
    'scipy',
    'matplotlib',
    'seaborn',
    'sklearn',
    'tensorflow',
    'netCDF4',
    'xarray',
    'powerlaw',
    'astropy',
    'jupyter',
    'pytest',
    'statsmodels',
    'tqdm',
    'dask'
]

def test_imports():
    """Test if all required packages can be imported."""
    failed_imports = []
    successful_imports = []
    
    for package in required_packages:
        try:
            # Handle special cases
            if package == 'sklearn':
                importlib.import_module('sklearn')
            else:
                importlib.import_module(package)
            successful_imports.append(package)
            print(f"✓ {package}")
        except ImportError as e:
            failed_imports.append((package, str(e)))
            print(f"✗ {package}: {e}")
    
    print(f"\nSummary:")
    print(f"Successful imports: {len(successful_imports)}")
    print(f"Failed imports: {len(failed_imports)}")
    
    if failed_imports:
        print("\nFailed packages:")
        for package, error in failed_imports:
            print(f"  - {package}: {error}")
        return False
    else:
        print("All packages imported successfully!")
        return True

if __name__ == "__main__":
    print("Testing package imports...")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print("-" * 50)
    
    success = test_imports()
    
    if success:
        print("\n🎉 Your environment is ready for solar flare analysis!")
    else:
        print("\n❌ Some packages are missing. Please install them using:")
        print("pip install -r requirements.txt")
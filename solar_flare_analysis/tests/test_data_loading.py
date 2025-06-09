#!/usr/bin/env python
"""
Test script to check data loading functionality
"""

import os
import sys
from src.data_processing.data_loader import GOESDataLoader

def test_data_loading():
    """Test the data loading functionality"""
    
    print("Testing GOES data loading...")
    
    # Initialize loader
    loader = GOESDataLoader()
    
    # Check available data files
    data_dir = 'data'
    print(f"\nData directory contents:")
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        print(f"  {file}: {size:.2f} MB")
    
    # Try to load a NetCDF file
    nc_files = [f for f in os.listdir(data_dir) if f.endswith('.nc')]
    if nc_files:
        test_file = os.path.join(data_dir, nc_files[0])
        print(f"\nLoading test file: {test_file}")
        
        try:
            data = loader.load_from_files([test_file])
            if data is not None:
                print(f"✓ Data loaded successfully!")
                print(f"  Shape: {data.shape}")
                print(f"  Columns: {list(data.columns)}")
                print(f"  Date range: {data.index.min()} to {data.index.max()}")
                print(f"  Sample data:")
                print(data.head())
            else:
                print("✗ Failed to load data (returned None)")
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No NetCDF files found in data directory")

if __name__ == "__main__":
    test_data_loading()

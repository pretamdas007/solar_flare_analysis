#!/usr/bin/env python3
"""
Debug script to check raw GOES data structure and content.
"""

import os
import sys
import numpy as np
import xarray as xr

# Add project root to path
sys.path.append(os.path.abspath('c:\\Users\\srabani\\Desktop\\goesflareenv\\solar_flare_analysis'))

from config import settings

def check_raw_data():
    """Check the raw GOES data file structure."""
    print("=== Checking Raw GOES Data ===")
    
    # Find data file
    sample_files = [f for f in os.listdir(settings.DATA_DIR) if f.endswith('.nc')]
    if not sample_files:
        print("No data files found!")
        return
        
    data_path = os.path.join(settings.DATA_DIR, sample_files[0])
    print(f"Checking: {data_path}")
    
    try:
        # Load raw data with xarray
        ds = xr.open_dataset(data_path)
        print(f"\nDataset structure:")
        print(ds)
        
        print(f"\nDataset variables:")
        for var in ds.data_vars:
            print(f"  {var}: {ds[var].shape} - {ds[var].dtype}")
            if hasattr(ds[var], 'long_name'):
                print(f"    Description: {ds[var].long_name}")
            if hasattr(ds[var], 'units'):
                print(f"    Units: {ds[var].units}")
            
            # Check for actual data
            data = ds[var].values
            if np.isfinite(data).any():
                print(f"    Valid data range: {np.nanmin(data):.2e} to {np.nanmax(data):.2e}")
                print(f"    Valid data points: {np.sum(np.isfinite(data))} / {len(data.flatten())}")
            else:
                print(f"    No valid data found (all NaN/Inf)")
        
        print(f"\nCoordinates:")
        for coord in ds.coords:
            print(f"  {coord}: {ds[coord].shape}")
            if coord == 'time' and len(ds[coord]) > 0:
                print(f"    Time range: {ds[coord].values[0]} to {ds[coord].values[-1]}")
        
        # Check specifically for XRS data
        xrs_vars = [var for var in ds.data_vars if 'xrs' in var.lower()]
        print(f"\nXRS variables found: {xrs_vars}")
        
        for var in xrs_vars:
            data = ds[var].values
            print(f"\n{var} detailed analysis:")
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            print(f"  Total elements: {data.size}")
            print(f"  Finite values: {np.sum(np.isfinite(data))}")
            print(f"  NaN values: {np.sum(np.isnan(data))}")
            print(f"  Infinite values: {np.sum(np.isinf(data))}")
            
            if np.isfinite(data).any():
                finite_data = data[np.isfinite(data)]
                print(f"  Min value: {np.min(finite_data):.2e}")
                print(f"  Max value: {np.max(finite_data):.2e}")
                print(f"  Mean value: {np.mean(finite_data):.2e}")
                print(f"  Sample values: {finite_data[:10]}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_raw_data()

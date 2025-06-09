#!/usr/bin/env python3
"""
Debug script to test ML input preparation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import netCDF4 as nc
from pathlib import Path

def debug_ml_input():
    """Debug the ML input preparation"""
    print("Debugging ML input preparation...")
    
    # Load some sample data
    data_dir = Path("data")
    data_files = list(data_dir.glob("*.nc"))
    
    if not data_files:
        print("No data files found")
        return
    
    filename = data_files[0]
    print(f"Using file: {filename}")
    
    # Load data like the backend does
    with nc.Dataset(str(filename), 'r') as dataset:
        times = dataset.variables['time'][:]
        xrs_a = dataset.variables['xrsa_flux'][:]
        xrs_b = dataset.variables['xrsb_flux'][:]
        
        # Convert time to datetime
        time_units = dataset.variables['time'].units
        base_time = pd.to_datetime('2000-01-01 12:00:00')
        timestamps = pd.to_datetime(times, unit='s', origin=base_time)
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'xrs_a': xrs_a,
            'xrs_b': xrs_b,
            'ratio': xrs_a / (xrs_b + 1e-12)
        })
        
        # Remove invalid data
        data = data.dropna()
        data = data[data['xrs_a'] > 0]
        data = data[data['xrs_b'] > 0]
        
        # Limit data points
        if len(data) > 10000:
            step = len(data) // 10000
            data = data.iloc[::step].reset_index(drop=True)
    
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {data.columns.tolist()}")
    print(f"XRS-A range: {data['xrs_a'].min():.2e} to {data['xrs_a'].max():.2e}")
    print(f"XRS-B range: {data['xrs_b'].min():.2e} to {data['xrs_b'].max():.2e}")
    
    # Test ML input preparation
    window_size = 100
    print(f"\nPreparing ML input with window_size={window_size}")
    
    # Normalize the data (log scale)
    xrs_a_log = np.log10(data['xrs_a'].values + 1e-12)
    xrs_b_log = np.log10(data['xrs_b'].values + 1e-12)
    
    print(f"XRS-A log range: {xrs_a_log.min():.2f} to {xrs_a_log.max():.2f}")
    print(f"XRS-B log range: {xrs_b_log.min():.2f} to {xrs_b_log.max():.2f}")
    
    # Create sliding windows
    if len(data) < window_size:
        pad_size = window_size - len(data)
        xrs_a_log = np.pad(xrs_a_log, (pad_size, 0), mode='edge')
        xrs_b_log = np.pad(xrs_b_log, (pad_size, 0), mode='edge')
        print(f"Padded data to {len(xrs_a_log)} points")
    
    # Take the last window_size points and create features that match model input (128 features)
    features_a = xrs_a_log[-window_size:]
    features_b = xrs_b_log[-window_size:]
    
    print(f"Features A shape: {features_a.shape}")
    print(f"Features B shape: {features_b.shape}")
    
    # Combine features to get exactly 128 dimensions
    if window_size >= 64:
        # Use last 64 points from each channel
        combined_features = np.concatenate([features_a[-64:], features_b[-64:]])
        print("Used last 64 points from each channel")
    else:
        # Pad to get 128 features total
        combined_features = np.concatenate([features_a, features_b])
        if len(combined_features) < 128:
            pad_size = 128 - len(combined_features)
            combined_features = np.pad(combined_features, (0, pad_size), mode='constant', constant_values=combined_features[-1])
            print(f"Padded combined features by {pad_size}")
        elif len(combined_features) > 128:
            combined_features = combined_features[:128]
            print("Truncated combined features to 128")
    
    print(f"Combined features shape: {combined_features.shape}")
    
    # Final reshape
    ml_input = combined_features.reshape(1, -1)
    print(f"Final ML input shape: {ml_input.shape}")
    
    # Test with different window sizes
    print("\nTesting different window sizes:")
    for ws in [50, 64, 100, 200]:
        features_a_test = xrs_a_log[-ws:] if len(xrs_a_log) >= ws else xrs_a_log
        features_b_test = xrs_b_log[-ws:] if len(xrs_b_log) >= ws else xrs_b_log
        
        if ws >= 64:
            combined_test = np.concatenate([features_a_test[-64:], features_b_test[-64:]])
        else:
            combined_test = np.concatenate([features_a_test, features_b_test])
            if len(combined_test) < 128:
                pad_size = 128 - len(combined_test)
                combined_test = np.pad(combined_test, (0, pad_size), mode='constant', constant_values=combined_test[-1])
            elif len(combined_test) > 128:
                combined_test = combined_test[:128]
        
        print(f"Window size {ws}: final shape {combined_test.reshape(1, -1).shape}")

if __name__ == "__main__":
    debug_ml_input()

#!/usr/bin/env python3

import pandas as pd
import os

def test_empty_csv():
    """Test what happens when we try to load an empty CSV file"""
    print("Testing empty CSV file loading...")
    
    csv_path = 'solar_flare_analysis/data/xrsb2_flux_observed.csv'
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"File {csv_path} does not exist")
        return
    
    # Check file size
    file_size = os.path.getsize(csv_path)
    print(f"File size: {file_size} bytes")
    
    # Check file contents
    with open(csv_path, 'r') as f:
        content = f.read()
        print(f"File content length: {len(content)} characters")
        print(f"File content: '{content[:100]}'")  # First 100 chars
    
    # Try to load with pandas
    try:
        df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
        print(f"Successfully loaded CSV:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Empty: {df.empty}")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        print(f"Error type: {type(e).__name__}")
        return None

if __name__ == "__main__":
    test_empty_csv()

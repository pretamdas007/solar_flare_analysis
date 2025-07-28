"""
Data Verification Script for Solar Flare Detection
This script checks your XRS data before training to ensure compatibility
"""

import os
import pandas as pd
import numpy as np
import glob
from datetime import datetime

def check_data_directory(data_dir):
    """
    Check the XRS data directory and files
    """
    print("🔍 CHECKING XRS DATA DIRECTORY")
    print("="*50)
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        return False
    
    print(f"✅ Directory found: {data_dir}")
    
    # Find CSV files
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    print(f"📁 Found {len(csv_files)} CSV files")
    
    if len(csv_files) == 0:
        print("❌ No CSV files found")
        return False
    
    # Check each file
    valid_files = []
    invalid_files = []
    
    for file in csv_files:
        print(f"\n📄 Checking: {os.path.basename(file)}")
        
        try:
            # Try to read the file
            df = pd.read_csv(file, nrows=5)  # Read only first 5 rows for checking
            
            print(f"   • Rows (sample): {len(df)}")
            print(f"   • Columns: {list(df.columns)}")
            
            # Check for required columns
            required_cols = ['time_tag', 'time_minutes', 'time_seconds']  # Accept multiple time formats
            flux_cols = [
                ['A_FLUX', 'B_FLUX'],                    # Standard format
                ['xrsa', 'xrsb'],                        # Alternative format
                ['xrsa_flux_observed', 'xrsb_flux_observed']  # Your actual format
            ]
            
            has_time = any(col in df.columns for col in required_cols)
            has_flux = any(all(col in df.columns for col in flux_set) for flux_set in flux_cols)
            
            if has_time and has_flux:
                print("   ✅ Valid file format")
                valid_files.append(file)
            else:
                print("   ❌ Missing required columns")
                if not has_time:
                    print("      Missing time column (time_tag, time_minutes, or time_seconds)")
                if not has_flux:
                    print("      Missing flux columns (A_FLUX/B_FLUX, xrsa/xrsb, or xrsa_flux_observed/xrsb_flux_observed)")
                invalid_files.append(file)
                
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
            invalid_files.append(file)
    
    print(f"\n📊 SUMMARY:")
    print(f"   ✅ Valid files: {len(valid_files)}")
    print(f"   ❌ Invalid files: {len(invalid_files)}")
    
    if valid_files:
        print(f"\n📋 Valid files by year:")
        year_files = {}
        for file in valid_files:
            filename = os.path.basename(file)
            # Try to extract year from filename
            for year in range(2020, 2030):
                if str(year) in filename:
                    if year not in year_files:
                        year_files[year] = []
                    year_files[year].append(filename)
                    break
        
        for year in sorted(year_files.keys()):
            print(f"   {year}: {len(year_files[year])} files")
            for file in year_files[year][:3]:  # Show first 3 files
                print(f"      - {file}")
            if len(year_files[year]) > 3:
                print(f"      ... and {len(year_files[year])-3} more")
    
    return len(valid_files) > 0

def preview_data_sample(data_dir):
    """
    Show a sample of the data
    """
    print(f"\n🔬 DATA PREVIEW")
    print("="*50)
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print("❌ No CSV files to preview")
        return
    
    # Take the first valid file
    for file in csv_files:
        try:
            df = pd.read_csv(file, nrows=10)
            print(f"📄 Sample from: {os.path.basename(file)}")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            
            # Show data types
            print(f"\n   Data Types:")
            for col in df.columns:
                print(f"      {col}: {df[col].dtype}")
            
            # Show sample data
            print(f"\n   Sample Data:")
            print(df.head(3).to_string(index=False))
            
            # Check for flux data
            flux_cols = []
            for col in df.columns:
                if any(flux_name in col.lower() for flux_name in ['flux', 'xrs']):
                    flux_cols.append(col)
            
            if flux_cols:
                print(f"\n   Flux Statistics:")
                for col in flux_cols[:2]:  # Show first 2 flux columns
                    values = pd.to_numeric(df[col], errors='coerce')
                    print(f"      {col}:")
                    print(f"         Min: {values.min():.2e}")
                    print(f"         Max: {values.max():.2e}")
                    print(f"         Mean: {values.mean():.2e}")
            
            break
            
        except Exception as e:
            print(f"❌ Error reading {file}: {e}")
            continue

def main():
    """
    Main data checking function
    """
    print("🌟 SOLAR FLARE DATA VERIFICATION")
    print("="*50)
    
    # Data directory
    data_dir = r'c:\Users\srabani\Desktop\goesflareenv\solar_flare_analysis\data\XRS'
    
    print(f"🎯 Target Directory: {data_dir}")
    
    # Check data
    is_valid = check_data_directory(data_dir)
    
    if is_valid:
        preview_data_sample(data_dir)
        
        print(f"\n✅ DATA CHECK PASSED!")
        print("🚀 Your data appears to be ready for training.")
        print("\n🎯 Next Steps:")
        print("   1. Run the training script: python train_model.py")
        print("   2. Monitor the training progress")
        print("   3. Check the generated visualizations")
        
    else:
        print(f"\n❌ DATA CHECK FAILED!")
        print("💡 Please fix the data issues before training:")
        print("   1. Ensure CSV files are in the correct directory")
        print("   2. Check that files have required columns:")
        print("      - time_tag, time_minutes, or time_seconds (datetime)")
        print("      - A_FLUX/B_FLUX, xrsa/xrsb, or xrsa_flux_observed/xrsb_flux_observed")
        print("   3. Verify data format is correct")

if __name__ == "__main__":
    main()

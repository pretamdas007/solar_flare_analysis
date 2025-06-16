#!/usr/bin/env python3
"""
XRS Data Loading Test Script
Tests the enhanced XRS data loading capabilities
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_xrs_data_loading():
    """
    Test XRS data loading and report findings
    """
    print("="*60)
    print("XRS DATA LOADING DIAGNOSTIC TEST")
    print("="*60)
    
    # Check data directory
    data_dir = Path("data/XRS")
    print(f"🔍 Checking data directory: {data_dir}")
    
    if not data_dir.exists():
        print(f"❌ Data directory does not exist: {data_dir}")
        print("📋 Available directories:")
        for item in Path("data").iterdir() if Path("data").exists() else []:
            if item.is_dir():
                print(f"   📁 {item.name}")
        return False
    
    # List files in directory
    all_files = list(data_dir.iterdir())
    csv_files = list(data_dir.glob("*.csv"))
    
    print(f"📁 Directory contents: {len(all_files)} total files")
    print(f"📄 CSV files found: {len(csv_files)}")
    
    if len(csv_files) == 0:
        print("❌ No CSV files found!")
        print("📋 Available files:")
        for file in all_files[:10]:  # Show first 10 files
            print(f"   📄 {file.name}")
        return False
    
    # Test loading first CSV file
    test_file = csv_files[0]
    print(f"\n🧪 Testing file: {test_file.name}")
    
    try:
        # Try different encodings
        df = None
        encoding_used = None
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                df = pd.read_csv(test_file, encoding=encoding)
                encoding_used = encoding
                print(f"✅ Successfully read with {encoding} encoding")
                break
            except UnicodeDecodeError:
                print(f"❌ Failed with {encoding} encoding")
                continue
        
        if df is None:
            print("❌ Could not read file with any encoding")
            return False
        
        # Analyze DataFrame
        print(f"📊 File analysis:")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Column names: {df.columns.tolist()}")
        
        # Check for XRS-like columns
        xrs_like_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if 'xrs' in col_lower or 'flux' in col_lower:
                xrs_like_columns.append(col)
        
        print(f"   XRS-like columns: {xrs_like_columns}")
        
        # Show sample data
        print(f"📋 Sample data (first 3 rows):")
        print(df.head(3))
        
        # Show data types
        print(f"📋 Data types:")
        for col in df.columns:
            print(f"   {col}: {df[col].dtype}")
        
        # Test our column mapping function
        print(f"\n🔧 Testing column standardization...")
        
        # Simple column mapping
        column_mappings = {
            'xrsa_flux_observed': 'xrs_a',
            'xrsb_flux_observed': 'xrs_b',
            'xrsa_flux': 'xrs_a',
            'xrsb_flux': 'xrs_b',
            'xrs_a': 'xrs_a',
            'xrs_b': 'xrs_b',
            'XRSA': 'xrs_a',
            'XRSB': 'xrs_b',
            'XRS_A': 'xrs_a',
            'XRS_B': 'xrs_b'
        }
        
        df_mapped = df.rename(columns=column_mappings)
        
        if 'xrs_a' not in df_mapped.columns or 'xrs_b' not in df_mapped.columns:
            print("❌ Standard mapping failed, trying pattern matching...")
            
            # Pattern matching
            xrs_a_candidates = [col for col in df.columns if 
                              ('xrs' in col.lower() and ('a' in col.lower() or '1' in col)) or
                              ('flux' in col.lower() and 'a' in col.lower())]
            xrs_b_candidates = [col for col in df.columns if 
                              ('xrs' in col.lower() and ('b' in col.lower() or '2' in col)) or
                              ('flux' in col.lower() and 'b' in col.lower())]
            
            print(f"   XRS-A candidates: {xrs_a_candidates}")
            print(f"   XRS-B candidates: {xrs_b_candidates}")
            
            if xrs_a_candidates and xrs_b_candidates:
                df_mapped['xrs_a'] = df[xrs_a_candidates[0]]
                df_mapped['xrs_b'] = df[xrs_b_candidates[0]]
                print(f"✅ Pattern matching successful!")
                print(f"   Mapped: {xrs_a_candidates[0]} -> xrs_a")
                print(f"   Mapped: {xrs_b_candidates[0]} -> xrs_b")
            else:
                print("❌ Pattern matching failed - no XRS columns found")
                return False
        else:
            print("✅ Standard column mapping successful!")
        
        # Analyze XRS data
        print(f"\n📈 XRS Data Analysis:")
        
        # Convert to numeric
        try:
            xrs_a = pd.to_numeric(df_mapped['xrs_a'], errors='coerce')
            xrs_b = pd.to_numeric(df_mapped['xrs_b'], errors='coerce')
            
            print(f"   XRS-A range: {xrs_a.min():.2e} to {xrs_a.max():.2e}")
            print(f"   XRS-B range: {xrs_b.min():.2e} to {xrs_b.max():.2e}")
            print(f"   XRS-A non-null: {xrs_a.count():,}/{len(xrs_a):,} ({xrs_a.count()/len(xrs_a)*100:.1f}%)")
            print(f"   XRS-B non-null: {xrs_b.count():,}/{len(xrs_b):,} ({xrs_b.count()/len(xrs_b)*100:.1f}%)")
            
            # Check for valid positive values
            valid_a = (xrs_a > 0) & (xrs_a < 1e-2) & (xrs_a > 1e-12)
            valid_b = (xrs_b > 0) & (xrs_b < 1e-2) & (xrs_b > 1e-12)
            
            print(f"   Valid XRS-A: {valid_a.sum():,}/{len(valid_a):,} ({valid_a.sum()/len(valid_a)*100:.1f}%)")
            print(f"   Valid XRS-B: {valid_b.sum():,}/{len(valid_b):,} ({valid_b.sum()/len(valid_b)*100:.1f}%)")
            
            # Test log transformation
            valid_both = valid_a & valid_b
            if valid_both.sum() > 0:
                print(f"   Valid both channels: {valid_both.sum():,} samples")
                
                # Apply log transformation
                xrs_a_log = np.log10(xrs_a[valid_both])
                xrs_b_log = np.log10(xrs_b[valid_both])
                
                print(f"   Log XRS-A range: {xrs_a_log.min():.2f} to {xrs_a_log.max():.2f}")
                print(f"   Log XRS-B range: {xrs_b_log.min():.2f} to {xrs_b_log.max():.2f}")
                
                # Create sample plot
                try:
                    plt.figure(figsize=(12, 8))
                    
                    plt.subplot(2, 2, 1)
                    plt.plot(xrs_a[valid_both][:1000], alpha=0.7, label='XRS-A')
                    plt.plot(xrs_b[valid_both][:1000], alpha=0.7, label='XRS-B')
                    plt.xlabel('Time Index')
                    plt.ylabel('Flux (W/m²)')
                    plt.title('Raw XRS Data (first 1000 points)')
                    plt.legend()
                    plt.yscale('log')
                    
                    plt.subplot(2, 2, 2)
                    plt.plot(xrs_a_log[:1000], alpha=0.7, label='XRS-A log')
                    plt.plot(xrs_b_log[:1000], alpha=0.7, label='XRS-B log')
                    plt.xlabel('Time Index')
                    plt.ylabel('Log10(Flux)')
                    plt.title('Log-transformed XRS Data')
                    plt.legend()
                    
                    plt.subplot(2, 2, 3)
                    plt.hist(xrs_a_log, bins=50, alpha=0.7, label='XRS-A log', density=True)
                    plt.hist(xrs_b_log, bins=50, alpha=0.7, label='XRS-B log', density=True)
                    plt.xlabel('Log10(Flux)')
                    plt.ylabel('Density')
                    plt.title('Log Flux Distribution')
                    plt.legend()
                    
                    plt.subplot(2, 2, 4)
                    plt.scatter(xrs_a_log[::100], xrs_b_log[::100], alpha=0.5, s=1)
                    plt.xlabel('XRS-A Log10(Flux)')
                    plt.ylabel('XRS-B Log10(Flux)')
                    plt.title('XRS-A vs XRS-B Correlation')
                    
                    plt.tight_layout()
                    plt.savefig('xrs_data_analysis.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"✅ Sample plot saved as 'xrs_data_analysis.png'")
                    
                except Exception as plot_error:
                    print(f"⚠ Could not create plot: {plot_error}")
                
                print(f"✅ XRS data loading test PASSED!")
                return True
            else:
                print("❌ No valid data found after filtering")
                return False
                
        except Exception as analysis_error:
            print(f"❌ Error in data analysis: {analysis_error}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing file: {e}")
        return False

def test_enhanced_loader():
    """
    Test the enhanced XRS data loader
    """
    print("\n" + "="*60)
    print("TESTING ENHANCED XRS DATA LOADER")
    print("="*60)
    
    try:
        # Import enhanced loader
        from enhanced_train_production import EnhancedXRSDataLoader
        
        # Initialize loader
        loader = EnhancedXRSDataLoader("data/XRS")
        
        # Load data
        print("🚀 Loading data with enhanced loader...")
        data = loader.load_and_process_xrs_data(max_files=2, sample_rate=0.5)
        
        if len(data) > 0:
            print(f"✅ Enhanced loader SUCCESS!")
            print(f"   Loaded shape: {data.shape}")
            print(f"   Data range: [{data.min():.3f}, {data.max():.3f}]")
            
            # Test sequence creation
            print("🔄 Testing sequence creation...")
            sequences, labels = loader.create_training_sequences(sequence_length=64)
            
            if len(sequences) > 0:
                print(f"✅ Sequence creation SUCCESS!")
                print(f"   Sequences shape: {sequences.shape}")
                print(f"   Labels shape: {labels.shape}")
                print(f"   Flare ratio: {np.mean(labels):.3f}")
                
                return True
            else:
                print("❌ No sequences created")
                return False
        else:
            print("❌ Enhanced loader failed - no data loaded")
            return False
            
    except ImportError:
        print("❌ Could not import enhanced loader")
        return False
    except Exception as e:
        print(f"❌ Enhanced loader test failed: {e}")
        return False

def main():
    """
    Run all XRS data tests
    """
    print("🚀 Starting XRS Data Loading Diagnostic Tests...\n")
    
    # Test 1: Basic data loading
    test1_result = test_xrs_data_loading()
    
    # Test 2: Enhanced loader
    test2_result = test_enhanced_loader()
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC TEST SUMMARY")
    print("="*60)
    
    print(f"📊 Basic XRS Loading Test: {'✅ PASSED' if test1_result else '❌ FAILED'}")
    print(f"🚀 Enhanced Loader Test: {'✅ PASSED' if test2_result else '❌ FAILED'}")
    
    if test1_result and test2_result:
        print("\n🎉 ALL TESTS PASSED! Your XRS data loading should work properly.")
        print("💡 You can now run the enhanced training script:")
        print("   python enhanced_train_production.py")
    elif test1_result:
        print("\n⚠ Basic loading works, but enhanced loader has issues.")
        print("💡 Try running the fixed training script:")
        print("   python train_production_fixed.py")
    else:
        print("\n❌ XRS data loading has fundamental issues.")
        print("💡 Possible solutions:")
        print("   1. Check if data/XRS directory exists")
        print("   2. Ensure CSV files are in the correct format")
        print("   3. Verify XRS column names in your CSV files")
        print("   4. Check file encodings (UTF-8 recommended)")

if __name__ == "__main__":
    main()

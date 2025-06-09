#!/usr/bin/env python3
"""
Debug script to analyze flare detection parameters and data characteristics.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath('c:\\Users\\srabani\\Desktop\\goesflareenv\\solar_flare_analysis'))

from config import settings
from src.data_processing.data_loader import load_goes_data, preprocess_xrs_data

def analyze_data_characteristics():
    """Analyze the characteristics of the GOES data to understand detection parameters."""
    print("=== Analyzing GOES Data Characteristics ===")
    
    # Load data
    sample_files = [f for f in os.listdir(settings.DATA_DIR) if f.endswith('.nc')]
    if not sample_files:
        print("No data files found!")
        return
        
    data_path = os.path.join(settings.DATA_DIR, sample_files[0])
    print(f"Analyzing: {data_path}")
    
    # Load and preprocess data
    data = load_goes_data(data_path)
    if data is None:
        print("Failed to load data")
        return
        
    df = preprocess_xrs_data(data, resample_freq='1min', apply_quality_filter=True, 
                            normalize=False, remove_background=False)
    
    # Focus on XRS-B channel
    flux_col = 'xrs_b'
    flux_data = df[flux_col].values
    
    print(f"\nData Statistics for {flux_col}:")
    print(f"  Total points: {len(flux_data)}")
    print(f"  Min value: {np.min(flux_data):.2e}")
    print(f"  Max value: {np.max(flux_data):.2e}")
    print(f"  Mean value: {np.mean(flux_data):.2e}")
    print(f"  Median value: {np.median(flux_data):.2e}")
    print(f"  Std deviation: {np.std(flux_data):.2e}")
    
    # Calculate percentiles
    percentiles = [90, 95, 99, 99.5, 99.9]
    print(f"\nPercentiles:")
    for p in percentiles:
        val = np.percentile(flux_data, p)
        print(f"  {p}th percentile: {val:.2e}")
    
    # Calculate rolling statistics for anomaly detection
    window_size = settings.DETECTION_PARAMS['window_size']
    rolling_mean = df[flux_col].rolling(window=window_size, center=True).mean()
    rolling_std = df[flux_col].rolling(window=window_size, center=True).std()
    
    # Current detection threshold
    threshold_factor = settings.DETECTION_PARAMS['threshold_factor']
    threshold = rolling_mean + threshold_factor * rolling_std
    
    print(f"\nCurrent Detection Parameters:")
    print(f"  Window size: {window_size}")
    print(f"  Threshold factor: {threshold_factor}")
    print(f"  Mean threshold: {np.nanmean(threshold):.2e}")
    print(f"  Max threshold: {np.nanmax(threshold):.2e}")
    
    # Find potential peaks with different thresholds
    print(f"\nPeak detection with different thresholds:")
    for factor in [1.0, 2.0, 3.0, 5.0, 10.0]:
        test_threshold = rolling_mean + factor * rolling_std
        peaks = df[flux_col] > test_threshold
        n_peaks = np.sum(peaks)
        print(f"  Factor {factor}: {n_peaks} potential peaks")
    
    # Plot the data and thresholds
    plt.figure(figsize=(15, 10))
    
    # Plot full time series (sample every 100 points for speed)
    sample_idx = slice(None, None, 100)
    times = df.index[sample_idx]
    flux_sample = df[flux_col].iloc[sample_idx]
    threshold_sample = threshold.iloc[sample_idx]
    
    plt.subplot(3, 1, 1)
    plt.plot(times, flux_sample, 'b-', alpha=0.7, label='XRS-B Flux')
    plt.plot(times, threshold_sample, 'r--', alpha=0.8, label=f'Threshold (factor={threshold_factor})')
    plt.yscale('log')
    plt.ylabel('Flux (W/m²)')
    plt.title('GOES XRS-B Time Series with Current Detection Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot histogram of flux values
    plt.subplot(3, 1, 2)
    plt.hist(flux_data, bins=100, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(flux_data), color='red', linestyle='--', label=f'Mean: {np.mean(flux_data):.2e}')
    plt.axvline(np.percentile(flux_data, 99), color='orange', linestyle='--', label=f'99th percentile: {np.percentile(flux_data, 99):.2e}')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Flux (W/m²)')
    plt.ylabel('Count')
    plt.title('Distribution of XRS-B Flux Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot zoomed-in view of high flux events
    plt.subplot(3, 1, 3)
    high_flux_threshold = np.percentile(flux_data, 95)
    high_flux_mask = df[flux_col] > high_flux_threshold
    
    if np.any(high_flux_mask):
        high_flux_periods = df[high_flux_mask]
        plt.plot(high_flux_periods.index, high_flux_periods[flux_col], 'ro', alpha=0.6, markersize=3)
        plt.yscale('log')
        plt.ylabel('Flux (W/m²)')
        plt.title(f'High Flux Events (>{high_flux_threshold:.2e} W/m²)')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No high flux events found', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('High Flux Events (None found)')
    
    plt.tight_layout()
    plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Recommend new thresholds
    print(f"\nRecommended Detection Parameters:")
    recommended_factor = 2.0  # More sensitive
    print(f"  Recommended threshold factor: {recommended_factor}")
    print(f"  This would detect approximately {np.sum(df[flux_col] > (rolling_mean + recommended_factor * rolling_std))} peaks")

if __name__ == "__main__":
    analyze_data_characteristics()

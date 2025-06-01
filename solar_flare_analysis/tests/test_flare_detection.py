#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for flare detection module.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.flare_detection.traditional_detection import (
    detect_flare_peaks, define_flare_bounds, detect_overlapping_flares
)


class TestFlareDetection(unittest.TestCase):
    """Test cases for traditional flare detection module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data with a single flare
        self.times = pd.date_range(
            start=datetime(2022, 1, 1), 
            end=datetime(2022, 1, 1, 1, 0), 
            freq='1min'
        )
        
        # Create a simple flare profile
        self.fluxes = np.array([
            1e-7, 1.2e-7, 1.5e-7, 2e-7, 3e-7,  # Background
            5e-7, 1e-6, 5e-6, 1e-5, 5e-6,     # Flare rising and peak
            1e-6, 5e-7, 3e-7, 2e-7, 1.5e-7    # Flare decay
        ])
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'time': self.times,
            'xrsb': self.fluxes
        })
        self.df.set_index('time', inplace=True)
        
        # Create sample data with two overlapping flares
        self.times_overlap = pd.date_range(
            start=datetime(2022, 1, 1), 
            end=datetime(2022, 1, 1, 2, 0), 
            freq='1min'
        )
        
        # First flare
        flare1 = np.array([
            1e-7, 1.2e-7, 1.5e-7, 2e-7, 3e-7,  # Background
            5e-7, 1e-6, 5e-6, 1e-5, 5e-6,     # Flare rising and peak
            1e-6, 5e-7, 3e-7, 2e-7, 1.5e-7    # Flare decay
        ])
        
        # Second flare starts before first one ends
        flare2 = np.zeros(len(self.times_overlap))
        flare2[10:25] = np.array([
            2e-7, 5e-7, 1e-6, 2e-6, 3e-6,     # Flare rising
            8e-6, 2e-5, 8e-6, 3e-6,           # Flare peak and initial decay
            1e-6, 5e-7, 3e-7, 2e-7, 1.5e-7    # Flare decay
        ])
        
        # Combine flares
        flux_overlap = flare1.copy()
        flux_overlap = np.append(flux_overlap, np.zeros(len(self.times_overlap) - len(flux_overlap)))
        flux_overlap += flare2
        
        # Create DataFrame with overlapping flares
        self.df_overlap = pd.DataFrame({
            'time': self.times_overlap,
            'xrsb': flux_overlap
        })
        self.df_overlap.set_index('time', inplace=True)
        
    def test_detect_flare_peaks(self):
        """Test flare peak detection."""
        # Detect peaks
        peaks = detect_flare_peaks(self.df, 'xrsb', threshold_factor=2, window_size=3)
        # Also test with float threshold
        peaks_float = detect_flare_peaks(self.df, 'xrsb', threshold_factor=2.0, window_size=3)
        self.assertEqual(len(peaks), len(peaks_float))
        self.assertEqual(peaks['peak_index'].iloc[0], peaks_float['peak_index'].iloc[0])        
        # Check that we found the expected peak
        self.assertEqual(len(peaks), 1)
        
        # Peak should be at the maximum value
        expected_peak_idx = np.argmax(self.fluxes)
        self.assertEqual(peaks['peak_index'].iloc[0], expected_peak_idx)
        
        # Check peak flux value
        self.assertEqual(peaks['peak_flux'].iloc[0], self.fluxes[expected_peak_idx])
    
    def test_define_flare_bounds(self):
        """Test defining flare boundaries."""
        # Detect peaks first
        peaks = detect_flare_peaks(self.df, 'xrsb', threshold_factor=2.0, window_size=3)
        
        # Define flare bounds
        flares = define_flare_bounds(
            self.df, 'xrsb', peaks['peak_index'].values,
            start_threshold=0.5, end_threshold=0.5,
            min_duration='1min', max_duration='1hour'
        )
        
        # Check that we defined bounds for the flare
        self.assertEqual(len(flares), 1)
        
        # Start index should be before peak
        self.assertLess(flares['start_index'].iloc[0], peaks['peak_index'].iloc[0])
        
        # End index should be after peak
        self.assertGreater(flares['end_index'].iloc[0], peaks['peak_index'].iloc[0])
        
        # Check other columns exist
        self.assertIn('start_time', flares.columns)
        self.assertIn('end_time', flares.columns)
        self.assertIn('duration', flares.columns)
    
    def test_detect_overlapping_flares(self):
        """Test detection of overlapping flares."""
        # Detect peaks in the overlapping flares data
        peaks = detect_flare_peaks(self.df_overlap, 'xrsb', threshold_factor=2, window_size=5)
        
        # We should detect both flare peaks
        self.assertEqual(len(peaks), 2)
        
        # Define flare bounds
        flares = define_flare_bounds(
            self.df_overlap, 'xrsb', peaks['peak_index'].values,
            start_threshold=0.5, end_threshold=0.5,
            min_duration='1min', max_duration='1hour'
        )
        
        # Detect overlapping flares
        overlapping = detect_overlapping_flares(flares, min_overlap='1min')
        
        # We should detect one pair of overlapping flares
        self.assertEqual(len(overlapping), 1)
        
        # Check the overlap duration
        self.assertIsInstance(overlapping[0][2], timedelta)


if __name__ == '__main__':
    unittest.main()

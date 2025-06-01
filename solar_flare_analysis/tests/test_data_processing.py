#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for data processing module.
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
from src.data_processing.data_loader import preprocess_xrs_data, remove_background


class TestDataLoader(unittest.TestCase):
    """Test cases for data processing module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.times = pd.date_range(
            start=datetime(2022, 1, 1), 
            end=datetime(2022, 1, 1, 1, 0), 
            freq='1min'
        )
        self.fluxes = np.array([
            1e-7, 1.2e-7, 1.5e-7, 2e-7, 3e-7,  # Background
            5e-7, 1e-6, 5e-6, 1e-5, 5e-6,     # Flare peak
            1e-6, 5e-7, 3e-7, 2e-7, 1.5e-7    # Return to background
        ])
        
        # Create mock DataFrame with quality flags
        self.df = pd.DataFrame({
            'time': self.times,
            'xrsa': self.fluxes * 0.5,  # A channel is typically lower
            'xrsb': self.fluxes,
            'xrsa_quality': np.zeros(len(self.times)),
            'xrsb_quality': np.zeros(len(self.times))
        })
        
        # Add some bad data points
        self.df.loc[3, 'xrsb_quality'] = 1  # Flag as bad
        self.df.loc[3, 'xrsb'] = np.nan     # Set to NaN
        
        # Create mock xarray dataset
        self.mock_data = {
            'time': self.times,
            'xrsa': ('time', self.fluxes * 0.5),
            'xrsb': ('time', self.fluxes),
            'xrsa_quality': ('time', np.zeros(len(self.times))),
            'xrsb_quality': ('time', np.zeros(len(self.times)))
        }
        self.mock_data['xrsb_quality'][1][3] = 1  # Flag as bad
        
    def test_remove_background(self):
        """Test background removal function."""
        # Apply background removal
        df_bg = remove_background(self.df, window_size=5, quantile=0.1)
        
        # Check that background columns were added
        self.assertIn('xrsa_background', df_bg.columns)
        self.assertIn('xrsb_background', df_bg.columns)
        self.assertIn('xrsa_no_background', df_bg.columns)
        self.assertIn('xrsb_no_background', df_bg.columns)
        
        # Check that background was removed (values should be lower)
        self.assertTrue((df_bg['xrsb_no_background'] <= df_bg['xrsb']).all())
        
        # Background should be lower than original signal
        self.assertTrue((df_bg['xrsb_background'] <= df_bg['xrsb']).all())
        
        # For simple case, check expected values
        expected_bg = np.min(self.fluxes)
        self.assertAlmostEqual(df_bg['xrsb_background'].min(), expected_bg, delta=expected_bg*0.5)
        
        # Peak flux should be preserved after background subtraction
        peak_idx = np.nanargmax(self.df['xrsb'])
        peak_flux = self.df['xrsb'].iloc[peak_idx]
        bg_flux = df_bg['xrsb_background'].iloc[peak_idx]
        expected_peak = peak_flux - bg_flux
        self.assertAlmostEqual(df_bg['xrsb_no_background'].iloc[peak_idx], expected_peak, delta=expected_peak*0.1)


if __name__ == '__main__':
    unittest.main()

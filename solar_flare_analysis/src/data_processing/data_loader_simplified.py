"""
Simplified data loading and preprocessing module for GOES XRS solar flare data
Only supports CSV file loading from the data folder with essential preprocessing functions
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
from scipy import signal

warnings.filterwarnings('ignore')


class GOESDataLoader:
    """
    Simplified GOES data loader for CSV files only
    """
    
    def __init__(self, data_dir='data'):
        """
        Initialize the GOES data loader
        
        Parameters
        ----------
        data_dir : str, optional
            Directory containing CSV data files (default: 'data')
        """
        self.data_dir = data_dir
        
        # Standard column mappings for XRS data
        self.column_mapping = {
            'time': ['time', 'timestamp', 'datetime', 'date_time'],
            'xrs_short': ['xrs_short', 'xrsa', 'short', 'channel_a', 'A_flux'],
            'xrs_long': ['xrs_long', 'xrsb', 'long', 'channel_b', 'B_flux']
        }
        
        # Data quality parameters
        self.quality_thresholds = {
            'min_flux': 1e-9,  # Minimum detectable flux
            'max_flux': 1e-3,  # Maximum reasonable flux
            'max_gap_minutes': 10,  # Maximum gap to interpolate
            'outlier_threshold': 5.0  # Sigma threshold for outlier detection
        }
    
    def load_csv_data(self, filename=None, date_range=None):
        """
        Load GOES data from CSV files in the data directory
        
        Parameters
        ----------
        filename : str, optional
            Specific CSV filename to load. If None, loads all CSV files in data_dir
        date_range : tuple of datetime, optional
            (start_date, end_date) to filter data
            
        Returns
        -------
        pd.DataFrame
            Loaded GOES XRS data with standardized columns
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory '{self.data_dir}' not found")
        
        if filename:
            # Load specific file
            filepath = os.path.join(self.data_dir, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File '{filepath}' not found")
            data = self._load_single_csv(filepath)
        else:
            # Load all CSV files in directory
            csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in '{self.data_dir}'")
            
            data_frames = []
            for csv_file in csv_files:
                filepath = os.path.join(self.data_dir, csv_file)
                try:
                    df = self._load_single_csv(filepath)
                    data_frames.append(df)
                except Exception as e:
                    print(f"Warning: Could not load {csv_file}: {e}")
            
            if not data_frames:
                raise ValueError("No valid CSV files could be loaded")
            
            # Combine all data
            data = pd.concat(data_frames, ignore_index=True)
            data = data.drop_duplicates().sort_values('time').reset_index(drop=True)
        
        # Filter by date range if provided
        if date_range:
            start_date, end_date = date_range
            data = data[(data['time'] >= start_date) & (data['time'] <= end_date)]
        
        return data
    
    def _load_single_csv(self, filepath):
        """
        Load and standardize a single CSV file
        
        Parameters
        ----------
        filepath : str
            Path to the CSV file
            
        Returns
        -------
        pd.DataFrame
            Standardized data
        """
        # Try different common CSV formats
        try:
            data = pd.read_csv(filepath)
        except Exception:
            # Try with different separators and encodings
            for sep in [',', ';', '\t']:
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                    try:
                        data = pd.read_csv(filepath, sep=sep, encoding=encoding)
                        break
                    except Exception:
                        continue
                else:
                    continue
                break
            else:
                raise ValueError(f"Could not read CSV file: {filepath}")
        
        # Standardize column names
        data = self._standardize_columns(data)
        
        # Convert time column to datetime
        if 'time' in data.columns:
            data['time'] = pd.to_datetime(data['time'], errors='coerce')
            data = data.dropna(subset=['time'])
            data = data.set_index('time').sort_index().reset_index()
        
        return data
    
    def _standardize_columns(self, data):
        """
        Standardize column names based on common patterns
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
            
        Returns
        -------
        pd.DataFrame
            Data with standardized column names
        """
        # Convert column names to lowercase for matching
        data.columns = data.columns.str.lower().str.strip()
        
        # Map columns to standard names
        new_columns = {}
        for standard_name, possible_names in self.column_mapping.items():
            for col in data.columns:
                if any(name in col for name in possible_names):
                    new_columns[col] = standard_name
                    break
        
        # Rename columns
        data = data.rename(columns=new_columns)
        
        # Ensure we have required columns
        required_cols = ['time', 'xrs_short', 'xrs_long']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}")
        
        return data
    
    def preprocess_data(self, data, resample_freq='1min', apply_quality_filter=True, 
                       normalize_channels=False, remove_background=False):
        """
        Preprocess GOES XRS data with essential functions
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        resample_freq : str, optional
            Resampling frequency (default: '1min')
        apply_quality_filter : bool, optional
            Whether to apply quality filters
        normalize_channels : bool, optional
            Whether to normalize channel data
        remove_background : bool, optional
            Whether to remove background trend
            
        Returns
        -------
        pd.DataFrame
            Preprocessed data
        """
        # Make a copy to avoid modifying original
        processed_data = data.copy()
        
        # Set time as index for processing
        if 'time' in processed_data.columns:
            processed_data = processed_data.set_index('time')
        
        # Apply quality filter
        if apply_quality_filter:
            processed_data = self._apply_quality_filter(processed_data)
        
        # Resample data
        if resample_freq:
            processed_data = self._resample_data(processed_data, resample_freq)
        
        # Handle missing values
        processed_data = self._handle_missing_values(processed_data)
        
        # Remove background trend
        if remove_background:
            processed_data = self._remove_background_trend(processed_data)
        
        # Normalize channels
        if normalize_channels:
            processed_data = self._normalize_channels(processed_data)
        
        # Reset index to have time as column
        processed_data = processed_data.reset_index()
        
        return processed_data
    
    def _apply_quality_filter(self, data):
        """Apply quality filters to remove invalid data"""
        for col in ['xrs_short', 'xrs_long']:
            if col in data.columns:
                # Remove values outside reasonable range
                mask = (data[col] >= self.quality_thresholds['min_flux']) & \
                       (data[col] <= self.quality_thresholds['max_flux'])
                data.loc[~mask, col] = np.nan
                
                # Remove outliers using z-score
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                data.loc[z_scores > self.quality_thresholds['outlier_threshold'], col] = np.nan
        
        return data
    
    def _resample_data(self, data, freq):
        """Resample data to specified frequency"""
        return data.resample(freq).mean()
    
    def _handle_missing_values(self, data):
        """Handle missing values with interpolation"""
        # Forward fill small gaps, then backward fill
        data = data.fillna(method='ffill', limit=self.quality_thresholds['max_gap_minutes'])
        data = data.fillna(method='bfill', limit=self.quality_thresholds['max_gap_minutes'])
        
        return data
    
    def _remove_background_trend(self, data, window_size='1H', quantile=0.1):
        """
        Remove background trend using rolling quantile
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        window_size : str, optional
            Window size for background estimation
        quantile : float, optional
            Quantile for background estimation
            
        Returns
        -------
        pd.DataFrame
            Data with background removed
        """
        data_detrended = data.copy()
        
        for col in ['xrs_short', 'xrs_long']:
            if col in data.columns:
                # Calculate rolling background
                background = data[col].rolling(window=window_size, center=True).quantile(quantile)
                background = background.fillna(method='ffill').fillna(method='bfill')
                
                # Subtract background
                data_detrended[col] = data[col] - background
                
                # Ensure no negative values
                data_detrended[col] = np.maximum(data_detrended[col], 
                                               data_detrended[col].min() * 0.1)
        
        return data_detrended
    
    def _normalize_channels(self, data):
        """Normalize channel data using log transformation"""
        for col in ['xrs_short', 'xrs_long']:
            if col in data.columns:
                # Apply log transformation
                data[col] = np.log10(data[col] + 1e-12)  # Add small value to avoid log(0)
        
        return data


# Backward compatibility wrapper functions
def load_goes_data(data_dir='data', filename=None, date_range=None, **kwargs):
    """
    Load GOES data from CSV files
    
    Parameters
    ----------
    data_dir : str, optional
        Directory containing CSV files
    filename : str, optional
        Specific filename to load
    date_range : tuple, optional
        Date range to filter data
    **kwargs
        Additional arguments (ignored for compatibility)
        
    Returns
    -------
    pd.DataFrame
        Loaded GOES XRS data
    """
    loader = GOESDataLoader(data_dir=data_dir)
    return loader.load_csv_data(filename=filename, date_range=date_range)


def preprocess_xrs_data(data, resample_freq='1min', apply_quality_filter=True, 
                       normalize=False, remove_background=False, **kwargs):
    """
    Preprocess XRS data with essential functions
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    resample_freq : str, optional
        Resampling frequency
    apply_quality_filter : bool, optional
        Whether to apply quality filters
    normalize : bool, optional
        Whether to normalize channels
    remove_background : bool, optional
        Whether to remove background trend
    **kwargs
        Additional arguments (ignored for compatibility)
        
    Returns
    -------
    pd.DataFrame
        Preprocessed data
    """
    loader = GOESDataLoader()
    return loader.preprocess_data(
        data, 
        resample_freq=resample_freq,
        apply_quality_filter=apply_quality_filter,
        normalize_channels=normalize,
        remove_background=remove_background
    )


def remove_background(data, window_size='1H', quantile=0.1, **kwargs):
    """
    Remove background trend from data
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    window_size : str, optional
        Window size for background estimation
    quantile : float, optional
        Quantile for background estimation
    **kwargs
        Additional arguments (ignored for compatibility)
        
    Returns
    -------
    pd.DataFrame
        Data with background removed
    """
    loader = GOESDataLoader()
    return loader._remove_background_trend(data, window_size=window_size, quantile=quantile)

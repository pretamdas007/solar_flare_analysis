"""
Data loading and preprocessing module for GOES XRS solar flare data.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import warnings
from datetime import datetime, timedelta

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

def load_goes_data(file_path):
    """
    Load GOES XRS data from a NetCDF (.nc) file.
    
    Parameters
    ----------
    file_path : str
        Path to the NetCDF file containing GOES XRS data
    
    Returns
    -------
    xarray.Dataset or None
        GOES XRS data as an xarray Dataset, None if loading fails
    """
    try:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
            
        data = xr.open_dataset(file_path)
        print(f"Successfully loaded data from {os.path.basename(file_path)}")
        print(f"Available variables: {list(data.data_vars)}")
        print(f"Time range: {data.time.min().values} to {data.time.max().values}")
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


def preprocess_xrs_data(data, channel='B', remove_bad_data=True, interpolate_gaps=True):
    """
    Preprocess GOES XRS data.
    
    Parameters
    ----------
    data : xarray.Dataset
        GOES XRS data as an xarray Dataset
    channel : str, optional
        XRS channel to process, 'A' (0.05-0.4 nm) or 'B' (0.1-0.8 nm)
    remove_bad_data : bool, optional
        If True, remove flagged bad data points
    interpolate_gaps : bool, optional
        If True, interpolate over data gaps
        
    Returns
    -------
    pandas.DataFrame or None
        Preprocessed XRS data with time index and flux values
    """
    try:
        if data is None:
            print("No data provided for preprocessing")
            return None
            
        # Extract the relevant channel data
        channel_lower = channel.lower()
        flux_var = f'xrs{channel_lower}'
        quality_var = f'xrs{channel_lower}_quality'
        
        # Try different variable name variations
        if flux_var in data.data_vars:
            flux = data[flux_var]
        elif f'xrs_{channel_lower}' in data.data_vars:
            flux = data[f'xrs_{channel_lower}']
        elif 'xrsb' in data.data_vars and channel_lower == 'b':
            flux = data['xrsb']
        elif 'xrsa' in data.data_vars and channel_lower == 'a':
            flux = data['xrsa']
        else:
            available_vars = list(data.data_vars)
            print(f"Channel {channel} not found. Available variables: {available_vars}")
            # Try to use the first XRS variable found
            xrs_vars = [var for var in available_vars if 'xrs' in var.lower() and 'quality' not in var.lower()]
            if xrs_vars:
                flux = data[xrs_vars[0]]
                print(f"Using {xrs_vars[0]} instead")
            else:
                raise ValueError(f"No XRS flux variables found in dataset")
        
        # Convert to pandas DataFrame
        df = flux.to_dataframe().reset_index()
        
        # Set time as index if present
        if 'time' in df.columns:
            df = df.set_index('time')
        
        # Get the flux column name (it might have changed during conversion)
        flux_cols = [col for col in df.columns if 'xrs' in col.lower() and 'quality' not in col.lower()]
        if not flux_cols:
            raise ValueError("No flux columns found after conversion")
        
        flux_col = flux_cols[0]
        
        # Handle quality flags if available
        if remove_bad_data:
            # Check for quality column
            quality_cols = [col for col in df.columns if 'quality' in col.lower() or 'flag' in col.lower()]
            if quality_cols:
                quality_col = quality_cols[0]
                print(f"Found quality column: {quality_col}")
                # Assuming 0 means good data
                bad_mask = df[quality_col] != 0
                df.loc[bad_mask, flux_col] = np.nan
                print(f"Flagged {bad_mask.sum()} bad data points as NaN")
        
        # Remove any infinite values
        df[flux_col] = df[flux_col].replace([np.inf, -np.inf], np.nan)
        
        # Interpolate over small gaps if requested
        if interpolate_gaps:
            original_nans = df[flux_col].isna().sum()
            df[flux_col] = df[flux_col].interpolate(method='time', limit=10)
            final_nans = df[flux_col].isna().sum()
            print(f"Interpolated {original_nans - final_nans} data points")
        
        # Remove remaining NaN values
        df = df.dropna(subset=[flux_col])
        
        # Check for log scale conversion
        if df[flux_col].median() < 0:
            print(f"Converting {flux_col} from log to linear scale")
            df[flux_col] = 10 ** df[flux_col]
        
        # Ensure positive values
        df[flux_col] = df[flux_col].clip(lower=1e-12)
        
        print(f"Preprocessed data shape: {df.shape}")
        print(f"Flux range: {df[flux_col].min():.2e} to {df[flux_col].max():.2e}")
        
        return df
        
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None


def extract_time_series(data, start_time=None, end_time=None, resample_freq=None):
    """
    Extract a specific time range from the GOES XRS data and optionally resample.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Preprocessed XRS data
    start_time : datetime or str, optional
        Start time for the extraction
    end_time : datetime or str, optional
        End time for the extraction
    resample_freq : str, optional
        Frequency for resampling (e.g., '1s', '60s')
        
    Returns
    -------
    pandas.DataFrame
        Extracted and resampled time series
    """
    if data is None or data.empty:
        print("No data provided for time series extraction")
        return None
        
    try:
        # Create a copy to avoid modifying the original
        df = data.copy()
        
        # Convert string times to datetime if necessary
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
        
        # Slice by time if specified
        if start_time is not None and end_time is not None:
            df = df.loc[start_time:end_time]
        elif start_time is not None:
            df = df.loc[start_time:]
        elif end_time is not None:
            df = df.loc[:end_time]
        
        # Resample if specified
        if resample_freq is not None:
            df = df.resample(resample_freq).mean()
        
        return df
        
    except Exception as e:
        print(f"Error extracting time series: {e}")
        return None


def remove_background(time_series, window_size=120, quantile=0.1):
    """
    Remove the background flux from a time series.
    
    Parameters
    ----------
    time_series : pandas.DataFrame
        Time series data
    window_size : int or str, optional
        Size of the rolling window for background estimation (in minutes if int)
    quantile : float, optional
        Quantile to use for background estimation within each window
        
    Returns
    -------
    pandas.DataFrame
        Time series with background removed
    """
    if time_series is None or time_series.empty:
        print("No data provided for background removal")
        return None
        
    try:
        # Create a copy to avoid modifying the original
        df = time_series.copy()
        
        # Convert window_size to pandas frequency string if it's an integer
        if isinstance(window_size, int):
            window_str = f'{window_size}min'
        else:
            window_str = window_size
        
        # Calculate the rolling background for each XRS column
        for col in df.columns:
            if col.startswith('xrs') and not col.endswith('_background') and not col.endswith('_no_background'):
                print(f"Removing background for column: {col}")
                
                # Calculate the rolling background using the specified quantile
                background = df[col].rolling(window=window_str, center=True).quantile(quantile)
                
                # Handle NaN values at the edges by using forward and backward fill
                background = background.bfill().ffill()
                
                # Store the background in a new column
                df[f'{col}_background'] = background
                
                # Subtract the background from the original signal
                df[f'{col}_no_background'] = df[col] - background
                
                # Ensure non-negative values (keep a small fraction of original for stability)
                min_flux = df[col] * 0.01  # 1% of original as minimum
                df[f'{col}_no_background'] = np.maximum(df[f'{col}_no_background'], min_flux)
        
        print("Background removal completed")
        return df
        
    except Exception as e:
        print(f"Error removing background: {e}")
        return time_series


def validate_data_quality(df, flux_col):
    """
    Validate the quality of preprocessed data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Preprocessed data
    flux_col : str
        Name of the flux column
        
    Returns
    -------
    dict
        Quality metrics
    """
    if df is None or df.empty:
        return {"valid": False, "reason": "Empty or None data"}
    
    try:
        quality_metrics = {
            "valid": True,
            "total_points": len(df),
            "missing_points": df[flux_col].isna().sum(),
            "data_coverage": 1 - (df[flux_col].isna().sum() / len(df)),
            "flux_range": (df[flux_col].min(), df[flux_col].max()),
            "mean_flux": df[flux_col].mean(),
            "time_span": df.index.max() - df.index.min(),
            "sampling_rate": df.index.to_series().diff().median()
        }
        
        # Check for common issues
        if quality_metrics["data_coverage"] < 0.8:
            quality_metrics["warnings"] = quality_metrics.get("warnings", [])
            quality_metrics["warnings"].append("Low data coverage (<80%)")
        
        if quality_metrics["flux_range"][0] <= 0:
            quality_metrics["warnings"] = quality_metrics.get("warnings", [])
            quality_metrics["warnings"].append("Non-positive flux values detected")
        
        return quality_metrics
        
    except Exception as e:
        return {"valid": False, "reason": f"Error validating data: {e}"}


# Test function to verify the module works
def test_data_loader():
    """Test function to verify the data loader works correctly."""
    print("Testing data_loader module...")
    
    # Test with mock data
    try:
        import xarray as xr
        import numpy as np
        
        # Create mock GOES XRS data
        time_range = pd.date_range('2023-01-01', periods=1440, freq='1min')
        mock_flux = 1e-7 + 1e-8 * np.random.random(len(time_range))
        
        mock_data = xr.Dataset({
            'xrsb': (['time'], mock_flux),
            'xrsb_quality': (['time'], np.zeros(len(time_range), dtype=int))
        }, coords={'time': time_range})
        
        print("✓ Mock data created successfully")
        
        # Test preprocessing
        df = preprocess_xrs_data(mock_data, channel='B')
        if df is not None:
            print("✓ Preprocessing successful")
            
            # Test background removal
            df_bg = remove_background(df)
            if df_bg is not None:
                print("✓ Background removal successful")
                
                # Test data quality validation
                quality = validate_data_quality(df, 'xrsb')
                if quality["valid"]:
                    print("✓ Data quality validation successful")
                    print("All tests passed!")
                    return True
        
        print("✗ Some tests failed")
        return False
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_data_loader()
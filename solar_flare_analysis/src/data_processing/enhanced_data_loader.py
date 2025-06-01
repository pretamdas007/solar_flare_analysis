"""
Enhanced data loading and preprocessing module for GOES XRS solar flare data
Supports multiple data sources and advanced preprocessing for ML analysis
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
from datetime import datetime, timedelta
import requests
import warnings
from urllib.parse import urljoin
from scipy import signal
from sklearn.preprocessing import StandardScaler, RobustScaler
import h5py

warnings.filterwarnings('ignore')


class GOESDataLoader:
    """
    Enhanced GOES data loader with multiple data source support
    """
    
    def __init__(self, cache_dir='data_cache'):
        """
        Initialize the GOES data loader
        
        Parameters
        ----------
        cache_dir : str, optional
            Directory for caching downloaded data
        """
        self.cache_dir = cache_dir
        self.base_urls = {
            'GOES-16': 'https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/',
            'GOES-17': 'https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/',
            'GOES-18': 'https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/'
        }
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Data quality parameters
        self.quality_thresholds = {
            'min_flux': 1e-9,  # Minimum detectable flux
            'max_flux': 1e-3,  # Maximum reasonable flux
            'max_gap_minutes': 10,  # Maximum gap to interpolate
            'outlier_threshold': 5.0  # Sigma threshold for outlier detection
        }
    
    def load_from_files(self, file_paths, start_date=None, end_date=None):
        """
        Load GOES data from local files
        
        Parameters
        ----------
        file_paths : str or list
            Path(s) to data files (NetCDF, CSV, or HDF5)
        start_date : str or datetime, optional
            Start date for data filtering
        end_date : str or datetime, optional
            End date for data filtering
            
        Returns
        -------
        pandas.DataFrame
            Loaded and preprocessed data
        """
        if isinstance(file_paths, str):
            if os.path.isdir(file_paths):
                # Load all files from directory
                file_paths = self._find_data_files(file_paths)
            else:
                file_paths = [file_paths]
        
        all_data = []
        
        for file_path in file_paths:
            try:
                if file_path.endswith('.nc'):
                    data = self._load_netcdf_file(file_path)
                elif file_path.endswith('.csv'):
                    data = self._load_csv_file(file_path)
                elif file_path.endswith(('.h5', '.hdf5')):
                    data = self._load_hdf5_file(file_path)
                else:
                    print(f"Unsupported file format: {file_path}")
                    continue
                
                if data is not None:
                    all_data.append(data)
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if not all_data:
            return None
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=False).sort_index()
        
        # Remove duplicates
        combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
        
        # Filter by date range if specified
        if start_date or end_date:
            combined_data = self._filter_by_date_range(combined_data, start_date, end_date)
        
        return combined_data
    
    def download_goes_data(self, start_date, end_date, satellite='GOES-16'):
        """
        Download GOES data from NOAA servers
        
        Parameters
        ----------
        start_date : str or datetime
            Start date for data download
        end_date : str or datetime
            End date for data download
        satellite : str, optional
            GOES satellite identifier
            
        Returns
        -------
        pandas.DataFrame
            Downloaded and preprocessed data
        """
        print(f"Downloading {satellite} data from {start_date} to {end_date}")
        
        # Convert dates to datetime objects
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Generate date range for download
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        all_data = []
        
        for date in date_range:
            try:
                daily_data = self._download_daily_data(date, satellite)
                if daily_data is not None:
                    all_data.append(daily_data)
            except Exception as e:
                print(f"Error downloading data for {date.strftime('%Y-%m-%d')}: {e}")
                continue
        
        if not all_data:
            print("No data downloaded")
            return None
        
        # Combine all daily data
        combined_data = pd.concat(all_data, ignore_index=False).sort_index()
        
        return combined_data
    
    def preprocess_data(self, data, resample_freq='1min', apply_quality_filter=True,
                       normalize_channels=False, remove_background=False):
        """
        Preprocess GOES XRS data for ML analysis
        
        Parameters
        ----------
        data : pandas.DataFrame
            Raw GOES data
        resample_freq : str, optional
            Resampling frequency (e.g., '1min', '30s')
        apply_quality_filter : bool, optional
            Whether to apply quality filtering
        normalize_channels : bool, optional
            Whether to normalize XRS channels
        remove_background : bool, optional
            Whether to remove background trend
            
        Returns
        -------
        pandas.DataFrame
            Preprocessed data
        """
        if data is None or len(data) == 0:
            return None
        
        processed_data = data.copy()
        
        # Apply quality filtering
        if apply_quality_filter:
            processed_data = self._apply_quality_filter(processed_data)
        
        # Resample data
        if resample_freq:
            processed_data = self._resample_data(processed_data, resample_freq)
        
        # Handle missing values
        processed_data = self._handle_missing_values(processed_data)
        
        # Remove background if requested
        if remove_background:
            processed_data = self._remove_background_trend(processed_data)
        
        # Normalize channels if requested
        if normalize_channels:
            processed_data = self._normalize_channels(processed_data)
        
        # Add derived features
        processed_data = self._add_derived_features(processed_data)
        
        print(f"Preprocessed data: {len(processed_data)} points from {processed_data.index[0]} to {processed_data.index[-1]}")
        
        return processed_data
    
    def _find_data_files(self, directory):
        """Find all data files in a directory"""
        supported_extensions = ['.nc', '.csv', '.h5', '.hdf5']
        files = []
        
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in supported_extensions):
                    files.append(os.path.join(root, filename))
        
        return sorted(files)
    
    def _load_netcdf_file(self, file_path):
        """Load data from NetCDF file"""
        try:
            with xr.open_dataset(file_path) as ds:
                # Extract time and XRS channels
                time = pd.to_datetime(ds.time.values)
                
                data_dict = {'time': time}
                
                # Look for XRS channels with different naming conventions
                xrs_vars = []
                for var in ds.data_vars:
                    if any(pattern in var.lower() for pattern in ['xrs', 'x-ray']):
                        xrs_vars.append(var)
                
                # Extract XRS data
                for var in xrs_vars:
                    if 'a' in var.lower() and 'long' not in var.lower():
                        data_dict['xrs_a'] = ds[var].values
                    elif 'b' in var.lower() and 'long' not in var.lower():
                        data_dict['xrs_b'] = ds[var].values
                
                if len(data_dict) > 1:  # More than just time
                    df = pd.DataFrame(data_dict)
                    df.set_index('time', inplace=True)
                    return df
                
        except Exception as e:
            print(f"Error loading NetCDF file {file_path}: {e}")
        
        return None
    
    def _load_csv_file(self, file_path):
        """Load data from CSV file"""
        try:
            df = pd.read_csv(file_path, parse_dates=True, index_col=0)
            
            # Standardize column names
            df.columns = [col.lower().replace('-', '_').replace(' ', '_') for col in df.columns]
            
            return df
            
        except Exception as e:
            print(f"Error loading CSV file {file_path}: {e}")
            return None
    
    def _load_hdf5_file(self, file_path):
        """Load data from HDF5 file"""
        try:
            with h5py.File(file_path, 'r') as f:
                # Extract time and data arrays
                time_data = f['time'][:]
                time = pd.to_datetime(time_data, unit='s') if time_data.dtype.kind in 'iu' else pd.to_datetime(time_data)
                
                data_dict = {'time': time}
                
                # Look for XRS data
                for key in f.keys():
                    if 'xrs' in key.lower():
                        data_dict[key.lower()] = f[key][:]
                
                if len(data_dict) > 1:
                    df = pd.DataFrame(data_dict)
                    df.set_index('time', inplace=True)
                    return df
                
        except Exception as e:
            print(f"Error loading HDF5 file {file_path}: {e}")
        
        return None
    
    def _filter_by_date_range(self, data, start_date, end_date):
        """Filter data by date range"""
        if start_date:
            start_date = pd.to_datetime(start_date)
            data = data[data.index >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            data = data[data.index <= end_date]
        
        return data
    
    def _download_daily_data(self, date, satellite):
        """Download data for a single day (placeholder for actual implementation)"""
        # This would implement actual download from NOAA servers
        # For now, return None to indicate no download capability
        print(f"Download functionality not implemented for {date.strftime('%Y-%m-%d')}")
        return None
    
    def _apply_quality_filter(self, data):
        """Apply quality filtering to remove bad data points"""
        filtered_data = data.copy()
        
        # Get XRS columns
        xrs_cols = [col for col in data.columns if 'xrs' in col.lower()]
        
        for col in xrs_cols:
            # Remove values outside reasonable range
            mask = (
                (filtered_data[col] >= self.quality_thresholds['min_flux']) &
                (filtered_data[col] <= self.quality_thresholds['max_flux']) &
                (~np.isnan(filtered_data[col])) &
                (~np.isinf(filtered_data[col]))
            )
            
            # Remove outliers using z-score
            z_scores = np.abs(signal.zscore(filtered_data[col].dropna()))
            outlier_mask = z_scores < self.quality_thresholds['outlier_threshold']
            
            # Combine masks
            final_mask = mask.copy()
            final_mask.loc[filtered_data[col].dropna().index] &= outlier_mask
            
            # Set bad values to NaN
            filtered_data.loc[~final_mask, col] = np.nan
        
        return filtered_data
    
    def _resample_data(self, data, freq):
        """Resample data to specified frequency"""
        try:
            # Use mean for resampling
            resampled = data.resample(freq).mean()
            return resampled
        except Exception as e:
            print(f"Error resampling data: {e}")
            return data
    
    def _handle_missing_values(self, data):
        """Handle missing values through interpolation and filling"""
        processed_data = data.copy()
        
        # Get XRS columns
        xrs_cols = [col for col in data.columns if 'xrs' in col.lower()]
        
        for col in xrs_cols:
            # Forward fill short gaps
            processed_data[col] = processed_data[col].fillna(method='ffill', limit=3)
            
            # Interpolate remaining gaps
            max_gap = self.quality_thresholds['max_gap_minutes']
            processed_data[col] = processed_data[col].interpolate(
                method='time', limit=max_gap
            )
            
            # Fill remaining NaNs with column median
            processed_data[col] = processed_data[col].fillna(processed_data[col].median())
        
        return processed_data
    
    def _remove_background_trend(self, data):
        """Remove background trend from XRS data"""
        processed_data = data.copy()
        
        xrs_cols = [col for col in data.columns if 'xrs' in col.lower()]
        
        for col in xrs_cols:
            # Use median filter to estimate background
            window_size = min(120, len(data) // 10)  # Adaptive window size
            if window_size > 3:
                background = signal.medfilt(processed_data[col].values, kernel_size=window_size)
                processed_data[f'{col}_detrended'] = processed_data[col] - background
                processed_data[f'{col}_background'] = background
        
        return processed_data
    
    def _normalize_channels(self, data):
        """Normalize XRS channels"""
        processed_data = data.copy()
        
        xrs_cols = [col for col in data.columns if 'xrs' in col.lower() and 'detrended' not in col]
        
        for col in xrs_cols:
            # Log transformation
            log_col = f'{col}_log'
            processed_data[log_col] = np.log10(processed_data[col].clip(lower=1e-12))
            
            # Z-score normalization
            norm_col = f'{col}_normalized'
            scaler = RobustScaler()
            processed_data[norm_col] = scaler.fit_transform(processed_data[[col]])
        
        return processed_data
    
    def _add_derived_features(self, data):
        """Add derived features for ML analysis"""
        processed_data = data.copy()
        
        xrs_cols = [col for col in data.columns if 'xrs' in col.lower() and 
                   not any(suffix in col for suffix in ['log', 'normalized', 'detrended', 'background'])]
        
        if len(xrs_cols) >= 2:
            # XRS-A to XRS-B ratio
            xrs_a_col = [col for col in xrs_cols if 'a' in col.lower()]
            xrs_b_col = [col for col in xrs_cols if 'b' in col.lower()]
            
            if xrs_a_col and xrs_b_col:
                ratio = processed_data[xrs_a_col[0]] / processed_data[xrs_b_col[0]].clip(lower=1e-12)
                processed_data['xrs_ratio'] = ratio.replace([np.inf, -np.inf], np.nan)
        
        # Add time-based features
        processed_data['hour'] = processed_data.index.hour
        processed_data['day_of_year'] = processed_data.index.dayofyear
        processed_data['is_weekend'] = processed_data.index.weekday >= 5
        
        # Add gradient features for each XRS channel
        for col in xrs_cols:
            # First derivative (rate of change)
            processed_data[f'{col}_gradient'] = np.gradient(processed_data[col].values)
            
            # Second derivative (acceleration)
            processed_data[f'{col}_acceleration'] = np.gradient(processed_data[f'{col}_gradient'].values)
            
            # Rolling statistics
            window = min(30, len(processed_data) // 10)
            if window > 1:
                processed_data[f'{col}_rolling_mean'] = processed_data[col].rolling(window).mean()
                processed_data[f'{col}_rolling_std'] = processed_data[col].rolling(window).std()
        
        return processed_data
    
    def create_ml_training_sequences(self, data, sequence_length=256, step_size=64, 
                                   target_columns=None, feature_columns=None):
        """
        Create sequences for ML training from time series data
        
        Parameters
        ----------
        data : pandas.DataFrame
            Preprocessed time series data
        sequence_length : int, optional
            Length of each sequence
        step_size : int, optional
            Step size between sequences
        target_columns : list, optional
            Columns to use as targets
        feature_columns : list, optional
            Columns to use as features
            
        Returns
        -------
        tuple
            X (features) and y (targets) arrays for ML training
        """
        if feature_columns is None:
            # Use XRS channels as default features
            feature_columns = [col for col in data.columns if 'xrs' in col.lower() and 
                             not any(suffix in col for suffix in ['log', 'normalized', 'background'])]
        
        if target_columns is None:
            # Use same as features for autoencoder-style training
            target_columns = feature_columns
        
        # Extract feature and target data
        X_data = data[feature_columns].values
        y_data = data[target_columns].values
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(0, len(data) - sequence_length + 1, step_size):
            X_seq = X_data[i:i + sequence_length]
            y_seq = y_data[i:i + sequence_length]
            
            # Skip sequences with too many NaN values
            if np.isnan(X_seq).sum() / X_seq.size < 0.1:  # Allow up to 10% NaN
                X_sequences.append(X_seq)
                y_sequences.append(y_seq)
        
        if len(X_sequences) == 0:
            return None, None
        
        X = np.array(X_sequences)
        y = np.array(y_sequences)
        
        # Handle any remaining NaN values
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        
        print(f"Created {len(X)} sequences of length {sequence_length}")
        
        return X, y
    
    def save_processed_data(self, data, filepath, format='hdf5'):
        """
        Save processed data to file
        
        Parameters
        ----------
        data : pandas.DataFrame
            Processed data to save
        filepath : str
            Output file path
        format : str, optional
            Output format ('hdf5', 'csv', 'parquet')
        """
        try:
            if format.lower() == 'hdf5':
                data.to_hdf(filepath, key='data', mode='w', complevel=9)
            elif format.lower() == 'csv':
                data.to_csv(filepath)
            elif format.lower() == 'parquet':
                data.to_parquet(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            print(f"Data saved to {filepath}")
            
        except Exception as e:
            print(f"Error saving data: {e}")


class SyntheticDataGenerator:
    """
    Generator for creating realistic synthetic solar flare data
    """
    
    def __init__(self, sampling_rate=60):  # 60 seconds sampling
        """
        Initialize synthetic data generator
        
        Parameters
        ----------
        sampling_rate : float, optional
            Sampling rate in seconds
        """
        self.sampling_rate = sampling_rate
        self.flare_templates = self._create_flare_templates()
    
    def generate_synthetic_dataset(self, duration_hours=24, flare_rate=0.1, 
                                 noise_level=0.1, include_nanoflares=True):
        """
        Generate a synthetic dataset with realistic flare characteristics
        
        Parameters
        ----------
        duration_hours : float, optional
            Duration of synthetic data in hours
        flare_rate : float, optional
            Average flare rate per hour
        noise_level : float, optional
            Noise level as fraction of signal
        include_nanoflares : bool, optional
            Whether to include nanoflares
            
        Returns
        -------
        pandas.DataFrame
            Synthetic GOES-like data
        """
        # Create time index
        n_points = int(duration_hours * 3600 / self.sampling_rate)
        time_index = pd.date_range('2023-01-01', periods=n_points, freq=f'{self.sampling_rate}s')
        
        # Initialize base signal (quiet Sun background)
        base_xrs_a = 1e-8 * np.ones(n_points)  # Quiet Sun background
        base_xrs_b = 1e-9 * np.ones(n_points)
        
        # Add gradual variations
        base_xrs_a *= (1 + 0.1 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 3600 / self.sampling_rate)))
        base_xrs_b *= (1 + 0.1 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 3600 / self.sampling_rate)))
        
        # Generate flare events
        expected_flares = int(duration_hours * flare_rate)
        flare_times = np.random.choice(n_points - 1000, expected_flares, replace=False)
        
        for flare_time in flare_times:
            # Choose flare class
            if include_nanoflares and np.random.random() < 0.6:
                flare_type = 'nanoflare'
            else:
                flare_type = np.random.choice(['A', 'B', 'C', 'M', 'X'], 
                                            p=[0.4, 0.3, 0.2, 0.08, 0.02])
            
            # Generate flare profile
            flare_profile_a, flare_profile_b = self._generate_flare_profile(flare_type, n_points)
            
            # Add to base signal
            start_idx = max(0, flare_time)
            end_idx = min(n_points, flare_time + len(flare_profile_a))
            
            if end_idx > start_idx:
                profile_slice = slice(0, end_idx - start_idx)
                signal_slice = slice(start_idx, end_idx)
                
                base_xrs_a[signal_slice] += flare_profile_a[profile_slice]
                base_xrs_b[signal_slice] += flare_profile_b[profile_slice]
        
        # Add noise
        noise_a = np.random.lognormal(0, noise_level, n_points) * base_xrs_a * 0.1
        noise_b = np.random.lognormal(0, noise_level, n_points) * base_xrs_b * 0.1
        
        xrs_a = base_xrs_a + noise_a
        xrs_b = base_xrs_b + noise_b
        
        # Create DataFrame
        synthetic_data = pd.DataFrame({
            'xrs_a': xrs_a,
            'xrs_b': xrs_b
        }, index=time_index)
        
        return synthetic_data
    
    def _create_flare_templates(self):
        """Create flare profile templates for different flare classes"""
        templates = {}
        
        # Nanoflare
        templates['nanoflare'] = {
            'peak_flux_a': (1e-10, 1e-9),
            'peak_flux_b': (1e-11, 1e-10),
            'rise_time': (60, 300),  # seconds
            'decay_time': (300, 1800)
        }
        
        # A-class
        templates['A'] = {
            'peak_flux_a': (1e-8, 1e-7),
            'peak_flux_b': (1e-9, 1e-8),
            'rise_time': (300, 900),
            'decay_time': (1800, 7200)
        }
        
        # B-class
        templates['B'] = {
            'peak_flux_a': (1e-7, 1e-6),
            'peak_flux_b': (1e-8, 1e-7),
            'rise_time': (600, 1800),
            'decay_time': (3600, 14400)
        }
        
        # C-class
        templates['C'] = {
            'peak_flux_a': (1e-6, 1e-5),
            'peak_flux_b': (1e-7, 1e-6),
            'rise_time': (900, 2700),
            'decay_time': (7200, 28800)
        }
        
        # M-class
        templates['M'] = {
            'peak_flux_a': (1e-5, 1e-4),
            'peak_flux_b': (1e-6, 1e-5),
            'rise_time': (1800, 5400),
            'decay_time': (14400, 57600)
        }
        
        # X-class
        templates['X'] = {
            'peak_flux_a': (1e-4, 1e-3),
            'peak_flux_b': (1e-5, 1e-4),
            'rise_time': (3600, 10800),
            'decay_time': (28800, 115200)
        }
        
        return templates
    
    def _generate_flare_profile(self, flare_type, max_length=1000):
        """Generate a realistic flare profile"""
        template = self.flare_templates[flare_type]
        
        # Sample parameters
        peak_flux_a = np.random.uniform(*template['peak_flux_a'])
        peak_flux_b = np.random.uniform(*template['peak_flux_b'])
        rise_time = np.random.uniform(*template['rise_time'])
        decay_time = np.random.uniform(*template['decay_time'])
        
        # Profile duration
        total_duration = rise_time + decay_time
        n_points = min(max_length, int(total_duration / self.sampling_rate))
        
        # Create time array
        t = np.arange(n_points) * self.sampling_rate
        peak_time = rise_time
        
        # Generate exponential rise and decay profile
        profile_a = np.zeros(n_points)
        profile_b = np.zeros(n_points)
        
        for i, time in enumerate(t):
            if time <= peak_time:
                # Rise phase - exponential approach to peak
                factor = 1 - np.exp(-time / (rise_time / 3))
            else:
                # Decay phase - exponential decay
                factor = np.exp(-(time - peak_time) / (decay_time / 3))
            
            profile_a[i] = peak_flux_a * factor
            profile_b[i] = peak_flux_b * factor
        
        return profile_a, profile_b

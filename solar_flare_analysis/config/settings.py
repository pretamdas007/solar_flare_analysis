"""
Configuration settings for solar flare analysis.
"""

import os

# Data settings
DATA_DIR = 'c:\\Users\\srabani\\Desktop\\goesflareenv\\solar_flare_analysis\\data'
OUTPUT_DIR = 'c:\\Users\\srabani\\Desktop\\goesflareenv\\solar_flare_analysis\\output'
MODEL_DIR = 'c:\\Users\\srabani\\Desktop\\goesflareenv\\solar_flare_analysis\\models'

# Default GOES XRS channels
XRS_CHANNELS = ['A', 'B']  # A: 0.05-0.4 nm(Higher-energy (shorter wavelength) soft X-rays), B: 0.1-0.8 nm(Lower-energy (longer wavelength) soft X-rays)

# Flare detection parameters
DETECTION_PARAMS = {
    'threshold_factor': 3.0,  # Standard deviations above moving median
    'window_size': 25,  # Window size for moving median
    'start_threshold': 0.01,  # Fraction of peak flux for start time
    'end_threshold': 0.5,  # Fraction of peak flux for end time
    'min_duration': '1min',  # Minimum flare duration
    'max_duration': '3H',  # Maximum flare duration
}

# Background removal parameters
BACKGROUND_PARAMS = {
    'window_size': '1H',  # Window size for background estimation
    'quantile': 0.1,  # Quantile for background estimation
}

# Machine Learning model parameters
ML_PARAMS = {
    'sequence_length': 128,  # Length of input time series
    'n_features': 1,  # Number of features per time point
    'max_flares': 3,  # Maximum number of overlapping flares to separate
    'dropout_rate': 0.2,  # Dropout rate for regularization
    'batch_size': 32,  # Training batch size
    'epochs': 100,  # Maximum training epochs
    'validation_split': 0.2,  # Fraction of data for validation
    'early_stopping_patience': 10,  # Patience for early stopping
}

# Power-law analysis parameters
POWERLAW_PARAMS = {
    'n_bootstrap': 1000,  # Number of bootstrap samples for uncertainty estimation
    'xmin': None,  # Minimum value for fitting, None for automatic estimation
    'xmax': None,  # Maximum value for fitting, None for no upper limit
}

# Visualization parameters
VISUALIZATION_PARAMS = {
    'default_figsize': (12, 8),  # Default figure size
    'dpi': 100,  # Figure resolution
    'cmap': 'viridis',  # Default colormap
    'save_format': 'png',  # Format for saving figures
}

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)

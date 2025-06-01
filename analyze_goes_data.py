"""
Simple GOES XRS data analysis script.
This standalone script analyzes GOES XRS netCDF files.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
from datetime import datetime, timedelta
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze GOES XRS data')
    parser.add_argument('--file', type=str, required=True,
                        help='Path to NetCDF data file')
    parser.add_argument('--output_dir', type=str, default='plots',
                        help='Directory to save output plots')
    return parser.parse_args()

def load_goes_data(filepath):
    """
    Load GOES XRS data from a NetCDF file.
    
    Parameters
    ----------
    filepath : str
        Path to the NetCDF file
        
    Returns
    -------
    dict
        Dictionary with time, XRS-A, and XRS-B data
    """
    print(f"Loading data from {filepath}...")
    
    try:
        # Open the NetCDF file
        dataset = nc.Dataset(filepath, 'r')
        
        # Extract time variable
        time_var = dataset.variables['time']
        
        # Convert time to datetime objects
        if hasattr(time_var, 'units'):
            time_units = time_var.units
            try:
                # Handle "seconds since" format
                if 'seconds since' in time_units:
                    base_time_str = time_units.replace('seconds since ', '')
                    base_time = datetime.strptime(base_time_str.strip(), '%Y-%m-%d %H:%M:%S')
                    times = np.array([base_time + timedelta(seconds=float(t)) for t in time_var[:]])
                else:
                    # Fallback for other formats
                    times = np.array([datetime.utcfromtimestamp(t) for t in time_var[:]])
            except:
                # Ultimate fallback
                print("Warning: Time conversion failed, using array indices")
                times = np.arange(len(time_var))
        else:
            # No time units specified
            times = np.arange(len(time_var))
        
        # Find XRS data variables
        xrsa_var = None
        xrsb_var = None
        
        for var_name in dataset.variables:
            if 'xrsa' in var_name.lower() and 'flux' in var_name.lower():
                xrsa_var = dataset.variables[var_name]
            elif 'xrsb' in var_name.lower() and 'flux' in var_name.lower():
                xrsb_var = dataset.variables[var_name]
        
        if xrsa_var is None or xrsb_var is None:
            # Try alternative variable names
            for var_name in dataset.variables:
                if ('xrs' in var_name.lower() or 'flux' in var_name.lower()) and 'a' in var_name.lower():
                    xrsa_var = dataset.variables[var_name]
                elif ('xrs' in var_name.lower() or 'flux' in var_name.lower()) and 'b' in var_name.lower():
                    xrsb_var = dataset.variables[var_name]
        
        # Extract flux data
        if xrsa_var is not None and xrsb_var is not None:
            xrsa_flux = xrsa_var[:]
            xrsb_flux = xrsb_var[:]
            
            # Check for quality flags
            xrsa_quality = None
            xrsb_quality = None
            
            for var_name in dataset.variables:
                if 'xrsa' in var_name.lower() and 'quality' in var_name.lower():
                    xrsa_quality = dataset.variables[var_name][:]
                elif 'xrsb' in var_name.lower() and 'quality' in var_name.lower():
                    xrsb_quality = dataset.variables[var_name][:]
            
            result = {
                'time': times,
                'xrsa': xrsa_flux,
                'xrsb': xrsb_flux
            }
            
            if xrsa_quality is not None:
                result['xrsa_quality'] = xrsa_quality
            if xrsb_quality is not None:
                result['xrsb_quality'] = xrsb_quality
            
            print(f"Loaded {len(times)} data points")
            return result
            
        else:
            print("Error: Could not find XRS flux variables")
            
            # List available variables
            print("Available variables:")
            for var_name in dataset.variables:
                print(f"  - {var_name}")
            
            return None
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    """
    Preprocess GOES XRS data.
    
    Parameters
    ----------
    data : dict
        Dictionary with raw data
        
    Returns
    -------
    pandas.DataFrame
        Preprocessed data with datetime index
    """
    # Create DataFrame with time index
    df = pd.DataFrame({
        'xrsa': data['xrsa'],
        'xrsb': data['xrsb']
    }, index=data['time'])
    
    # Add quality flags if available
    if 'xrsa_quality' in data:
        df['xrsa_quality'] = data['xrsa_quality']
    if 'xrsb_quality' in data:
        df['xrsb_quality'] = data['xrsb_quality']
    
    # Remove bad data points if quality flags are available
    if 'xrsa_quality' in df.columns:
        df.loc[df['xrsa_quality'] > 0, 'xrsa'] = np.nan
    if 'xrsb_quality' in df.columns:
        df.loc[df['xrsb_quality'] > 0, 'xrsb'] = np.nan
    
    # Interpolate small gaps
    df['xrsa'] = df['xrsa'].interpolate(method='linear', limit=10)
    df['xrsb'] = df['xrsb'].interpolate(method='linear', limit=10)
    
    return df

def remove_background(df, window_size=60, quantile=0.1):
    """
    Remove background from XRS data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with XRS data
    window_size : int
        Rolling window size in minutes
    quantile : float
        Quantile to use for background estimation
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with background and background-subtracted flux
    """
    # Create a copy of the DataFrame
    df_result = df.copy()
    
    # Calculate rolling quantile as background
    for channel in ['xrsa', 'xrsb']:
        if channel in df.columns:
            # Compute background as rolling quantile
            df_result[f'{channel}_background'] = df[channel].rolling(
                window=window_size, center=False
            ).quantile(quantile)
            
            # Forward fill NaN values
            df_result[f'{channel}_background'] = df_result[f'{channel}_background'].fillna(method='ffill')
            
            # Backward fill any remaining NaNs
            df_result[f'{channel}_background'] = df_result[f'{channel}_background'].fillna(method='bfill')
            
            # Subtract background
            df_result[f'{channel}_bg_subtracted'] = df[channel] - df_result[f'{channel}_background']
            
            # Ensure no negative values
            df_result[f'{channel}_bg_subtracted'] = df_result[f'{channel}_bg_subtracted'].clip(lower=0)
    
    return df_result

def detect_flares(df, flux_col, threshold_factor=3, window_size=5):
    """
    Detect solar flares in XRS data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with XRS data
    flux_col : str
        Column name for flux data
    threshold_factor : float
        Factor above background to consider a flare
    window_size : int
        Window size for peak detection
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with flare information
    """
    # Calculate threshold from background
    if f'{flux_col}_background' in df.columns:
        background = df[f'{flux_col}_background']
    else:
        background = df[flux_col].rolling(window=180, center=True).quantile(0.1)
        background = background.fillna(method='ffill').fillna(method='bfill')
    
    threshold = background * threshold_factor
    
    # Find potential peaks
    peak_indices = []
    for i in range(window_size, len(df) - window_size):
        window = df.iloc[i-window_size:i+window_size+1]
        center = df.iloc[i]
        
        if (center[flux_col] > threshold.iloc[i] and 
            center[flux_col] == window[flux_col].max()):
            peak_indices.append(i)
    
    # Build flare information
    flares = []
    
    for peak_idx in peak_indices:
        # Get peak properties
        peak_time = df.index[peak_idx]
        peak_flux = df.iloc[peak_idx][flux_col]
        
        # Define flare bounds
        start_idx = peak_idx
        while start_idx > 0:
            current = df.iloc[start_idx][flux_col]
            prev = df.iloc[start_idx-1][flux_col]
            
            if current <= background.iloc[start_idx] * 1.2 or current > prev:
                break
            start_idx -= 1
        
        end_idx = peak_idx
        while end_idx < len(df) - 1:
            current = df.iloc[end_idx][flux_col]
            next_val = df.iloc[end_idx+1][flux_col]
            
            if current <= background.iloc[end_idx] * 1.2 or current < next_val:
                break
            end_idx += 1
        
        # Calculate flare duration
        if hasattr(df.index[0], 'timestamp'):
            start_time = df.index[start_idx]
            end_time = df.index[end_idx]
            duration_minutes = (end_time - start_time).total_seconds() / 60
        else:
            duration_minutes = end_idx - start_idx
        
        # Add to flares list if duration is reasonable
        if 1 <= duration_minutes <= 180:  # Between 1 and 180 minutes
            flare_info = {
                'start_time': df.index[start_idx],
                'peak_time': peak_time,
                'end_time': df.index[end_idx],
                'peak_flux': peak_flux,
                'duration_minutes': duration_minutes
            }
            flares.append(flare_info)
    
    return pd.DataFrame(flares)

def classify_flare(peak_flux):
    """
    Classify a solar flare based on its peak flux.
    
    Parameters
    ----------
    peak_flux : float
        Peak flux in W/m²
        
    Returns
    -------
    str
        Flare classification (A, B, C, M, or X)
    """
    if peak_flux < 1e-7:
        return f"A{peak_flux/1e-8:.1f}"
    elif peak_flux < 1e-6:
        return f"B{peak_flux/1e-7:.1f}"
    elif peak_flux < 1e-5:
        return f"C{peak_flux/1e-6:.1f}"
    elif peak_flux < 1e-4:
        return f"M{peak_flux/1e-5:.1f}"
    else:
        return f"X{peak_flux/1e-4:.1f}"

def plot_xrs_data(df, flares=None, output_dir='plots'):
    """
    Plot XRS data with detected flares.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with XRS data
    flares : pandas.DataFrame, optional
        DataFrame with flare information
    output_dir : str
        Directory to save output plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot GOES XRS-A data
    if 'xrsa' in df.columns:
        axs[0].plot(df.index, df['xrsa'], 'b-', label='XRS-A (0.05-0.4 nm)')
        if 'xrsa_background' in df.columns:
            axs[0].plot(df.index, df['xrsa_background'], 'b--', alpha=0.5, label='Background')
        axs[0].set_yscale('log')
        axs[0].set_ylabel('Irradiance (W/m²)')
        axs[0].set_title('GOES XRS-A (0.05-0.4 nm)')
        axs[0].grid(True, which='both', linestyle='--', alpha=0.7)
        axs[0].legend()
    
    # Plot GOES XRS-B data
    if 'xrsb' in df.columns:
        axs[1].plot(df.index, df['xrsb'], 'r-', label='XRS-B (0.1-0.8 nm)')
        if 'xrsb_background' in df.columns:
            axs[1].plot(df.index, df['xrsb_background'], 'r--', alpha=0.5, label='Background')
        axs[1].set_yscale('log')
        axs[1].set_ylabel('Irradiance (W/m²)')
        axs[1].set_title('GOES XRS-B (0.1-0.8 nm)')
        axs[1].grid(True, which='both', linestyle='--', alpha=0.7)
        axs[1].legend()
    
    # Mark flare classes
    for ax in axs:
        ax.axhline(y=1e-4, color='red', linestyle='-', alpha=0.3)
        ax.axhline(y=1e-5, color='orange', linestyle='-', alpha=0.3)
        ax.axhline(y=1e-6, color='green', linestyle='-', alpha=0.3)
        ax.axhline(y=1e-7, color='blue', linestyle='-', alpha=0.3)
        ax.text(df.index[0], 5e-4, 'X', color='red')
        ax.text(df.index[0], 5e-5, 'M', color='orange')
        ax.text(df.index[0], 5e-6, 'C', color='green')
        ax.text(df.index[0], 5e-7, 'B', color='blue')
        ax.text(df.index[0], 5e-8, 'A', color='purple')
    
    # Mark detected flares
    if flares is not None and len(flares) > 0:
        for _, flare in flares.iterrows():
            # XRS-B plot
            if 'xrsb' in df.columns:
                axs[1].axvline(x=flare['peak_time'], color='red', linestyle='--', alpha=0.7)
                axs[1].axvspan(flare['start_time'], flare['end_time'], alpha=0.2, color='yellow')
                
                # Annotate flare class if peak_flux is available
                if 'peak_flux' in flare:
                    flare_class = classify_flare(flare['peak_flux'])
                    axs[1].annotate(flare_class, 
                                  xy=(flare['peak_time'], flare['peak_flux']),
                                  xytext=(0, 10),
                                  textcoords='offset points',
                                  arrowprops=dict(arrowstyle='->'),
                                  fontsize=8)
    
    # Format x-axis
    fig.autofmt_xdate()
    plt.tight_layout()
    
    # Save figure
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f'goes_xrs_analysis_{timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    # Display figure
    plt.show()

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Load data
    data = load_goes_data(args.file)
    
    if data is None:
        print("Error loading data. Exiting.")
        return
    
    # Preprocess data
    df = preprocess_data(data)
    
    # Remove background
    df_bg = remove_background(df)
    
    # Detect flares
    flares_b = detect_flares(df_bg, 'xrsb')
    
    # Print flare information
    if len(flares_b) > 0:
        print(f"\nDetected {len(flares_b)} flares in XRS-B channel:")
        for i, flare in flares_b.iterrows():
            flare_class = classify_flare(flare['peak_flux'])
            print(f"  {i+1}. {flare_class} flare at {flare['peak_time']}, "
                 f"duration: {flare['duration_minutes']:.1f} minutes")
    else:
        print("\nNo flares detected in XRS-B channel")
    
    # Plot data
    plot_xrs_data(df_bg, flares_b, args.output_dir)

if __name__ == "__main__":
    main()

"""
Traditional flare detection algorithms for solar X-ray data.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def detect_flare_peaks(time_series, flux_column, threshold_factor=3, window_size=25):
    """
    Detect peaks in the time series that could represent flares.
    
    Parameters
    ----------
    time_series : pandas.DataFrame
        Time series data containing flux measurements
    flux_column : str
        Name of the column containing flux data
    threshold_factor : float, optional
        Factor multiplied with the standard deviation to set the detection threshold
    window_size : int or str, optional
        Size of the window for peak detection. Can be integer or time string like '10min'
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the detected peak indices and times
    """
    # Calculate the moving median and standard deviation
    flux = time_series[flux_column].values
    time_index = time_series.index
    
    # Convert window_size to integer if it's a string (time-based)
    if isinstance(window_size, str):
        # Convert time-based window to number of samples
        # Assuming 1-minute sampling, convert minutes to samples
        if 'min' in window_size.lower():
            window_samples = int(window_size.lower().replace('min', ''))
        elif 'h' in window_size.lower():
            window_samples = int(window_size.lower().replace('h', '')) * 60
        else:
            window_samples = int(window_size)
    else:
        window_samples = int(window_size)
    
    # Ensure window size is odd and at least 3
    if window_samples % 2 == 0:
        window_samples += 1
    window_samples = max(window_samples, 3)    # Use scipy.signal.find_peaks for peak detection
    median = signal.medfilt(flux, kernel_size=window_samples)
    std_dev = np.std(flux)
    threshold = median + threshold_factor * std_dev
    
    # Debug output
    print(f"  Window samples: {window_samples}")
    print(f"  Flux range: {np.min(flux):.2e} to {np.max(flux):.2e}")
    print(f"  Median flux: {np.mean(median):.2e}")
    print(f"  Std dev: {std_dev:.2e}")
    print(f"  Threshold range: {np.min(threshold):.2e} to {np.max(threshold):.2e}")
    print(f"  Flux above threshold: {np.sum(flux > threshold)} points")
    
    # Find peaks above the threshold
    peaks, properties = signal.find_peaks(
        flux, 
        height=threshold,
        distance=window_samples // 2,  # Minimum distance between peaks
        prominence=std_dev,  # Minimum prominence for a peak
        width=3  # Minimum width for a peak
    )
    
    # Create a DataFrame with the detected peaks
    peak_data = {
        'peak_index': peaks,
        'peak_time': [time_index[i] for i in peaks],
        'peak_flux': [flux[i] for i in peaks],
        'prominence': properties['prominences'],
        'width': properties['widths']
    }
    
    peak_df = pd.DataFrame(peak_data)
    return peak_df


def define_flare_bounds(time_series, flux_column, peak_indices, 
                        start_threshold=0.01, end_threshold=0.5,
                        min_duration='1min', max_duration='3H'):
    """
    Define the start and end times of flares based on detected peaks.
    
    Parameters
    ----------
    time_series : pandas.DataFrame
        Time series data containing flux measurements
    flux_column : str
        Name of the column containing flux data
    peak_indices : list
        List of indices corresponding to detected flare peaks
    start_threshold : float, optional
        Fraction of peak flux to define flare start
    end_threshold : float, optional
        Fraction of peak flux to define flare end
    min_duration : str, optional
        Minimum duration for a valid flare
    max_duration : str, optional
        Maximum duration for a valid flare
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing start and end times for each flare
    """
    flux = time_series[flux_column].values
    time_index = time_series.index
    
    min_dur = pd.Timedelta(min_duration)
    max_dur = pd.Timedelta(max_duration)
    
    flares = []
    
    for peak_idx in peak_indices:
        peak_flux = flux[peak_idx]
        
        # Find start time (going backward from peak)
        start_idx = peak_idx
        while (start_idx > 0 and 
               flux[start_idx] > peak_flux * start_threshold):
            start_idx -= 1
        
        # Find end time (going forward from peak)
        end_idx = peak_idx
        while (end_idx < len(flux) - 1 and 
               flux[end_idx] > peak_flux * end_threshold):
            end_idx += 1
        
        # Check if the flare duration is within allowed limits
        duration = time_index[end_idx] - time_index[start_idx]
        
        if min_dur <= duration <= max_dur:
            flare_info = {
                'peak_index': peak_idx,
                'start_index': start_idx,
                'end_index': end_idx,
                'start_time': time_index[start_idx],
                'peak_time': time_index[peak_idx],
                'end_time': time_index[end_idx],
                'duration': duration,
                'peak_flux': peak_flux,                'integrated_flux': np.trapezoid(
                    flux[start_idx:end_idx+1],
                    (time_index[start_idx:end_idx+1] - time_index[start_idx]).total_seconds()
                )
            }
            flares.append(flare_info)
    
    return pd.DataFrame(flares)


def fit_flare_profile(time_series, flux_column, start_idx, peak_idx, end_idx, plot=False):
    """
    Fit a parametric model to a flare profile.
    
    Parameters
    ----------
    time_series : pandas.DataFrame
        Time series data containing flux measurements
    flux_column : str
        Name of the column containing flux data
    start_idx : int
        Index corresponding to flare start
    peak_idx : int
        Index corresponding to flare peak
    end_idx : int
        Index corresponding to flare end
    plot : bool, optional
        If True, plot the flare and the fitted profile
        
    Returns
    -------
    dict
        Dictionary containing the fitted parameters
    """
    flux = time_series[flux_column].values[start_idx:end_idx+1]
    times = np.arange(len(flux))  # Use array indices for fitting
    
    # Define a simple flare model (exponential rise, exponential decay)
    def flare_model(t, a, t_peak, rise_tau, decay_tau, background):
        rise = np.exp(-(t_peak - t) / rise_tau) * (t <= t_peak)
        decay = np.exp(-(t - t_peak) / decay_tau) * (t > t_peak)
        return a * (rise + decay) + background
    
    # Initial parameter estimates
    p0 = [
        flux[peak_idx - start_idx] - min(flux),  # amplitude
        peak_idx - start_idx,  # peak time index
        (peak_idx - start_idx) / 3,  # rise time constant
        (end_idx - peak_idx) / 3,  # decay time constant
        min(flux)  # background
    ]
    
    try:
        # Fit the model
        popt, pcov = curve_fit(flare_model, times, flux, p0=p0, maxfev=5000)
        
        # Calculate fit quality
        fitted_flux = flare_model(times, *popt)
        residuals = flux - fitted_flux
        sse = np.sum(residuals**2)
        rmse = np.sqrt(np.mean(residuals**2))
        
        # Calculate parameter errors
        perr = np.sqrt(np.diag(pcov))
        
        # If requested, plot the flare and the fitted model
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(times, flux, 'b-', label='Original Flare')
            plt.plot(times, fitted_flux, 'r--', label='Fitted Model')
            plt.title('Flare Profile and Fitted Model')
            plt.xlabel('Time Index')
            plt.ylabel('Flux')
            plt.legend()
            plt.grid(True)
            plt.show()
        
        # Return the fitted parameters and fit quality
        return {
            'amplitude': popt[0],
            'peak_index': popt[1] + start_idx,
            'rise_tau': popt[2],
            'decay_tau': popt[3],
            'background': popt[4],
            'sse': sse,
            'rmse': rmse,
            'param_errors': perr
        }
    
    except Exception as e:
        print(f"Fitting error: {e}")
        return None


def detect_overlapping_flares(flares_df, min_overlap='2min'):
    """
    Detect potentially overlapping flares based on their time bounds.
    
    Parameters
    ----------
    flares_df : pandas.DataFrame
        DataFrame containing flare information with start_time and end_time
    min_overlap : str, optional
        Minimum overlap duration to consider flares as overlapping
        
    Returns
    -------
    list
        List of tuples containing pairs of overlapping flare indices    """
    overlapping_pairs = []
    
    # Check if DataFrame is empty or doesn't have required columns
    if flares_df.empty or 'start_time' not in flares_df.columns or 'end_time' not in flares_df.columns:
        return overlapping_pairs
    
    min_overlap_duration = pd.Timedelta(min_overlap)
    
    # Sort by start time to optimize comparison
    flares_sorted = flares_df.sort_values('start_time').reset_index(drop=True)
    
    # Compare each flare with subsequent ones
    for i in range(len(flares_sorted) - 1):
        for j in range(i + 1, len(flares_sorted)):
            flare_i = flares_sorted.iloc[i]
            flare_j = flares_sorted.iloc[j]
            
            # If the second flare starts after the first one ends, 
            # no need to check further pairs
            if flare_j['start_time'] >= flare_i['end_time']:
                break
                
            # Calculate overlap duration
            overlap_start = max(flare_i['start_time'], flare_j['start_time'])
            overlap_end = min(flare_i['end_time'], flare_j['end_time'])
            overlap_duration = overlap_end - overlap_start
            
            # Check if overlap exceeds minimum duration
            if overlap_duration >= min_overlap_duration:
                # Get original indices from the input DataFrame
                original_i = flares_df.index[flares_df['start_time'] == flare_i['start_time']].tolist()[0]
                original_j = flares_df.index[flares_df['start_time'] == flare_j['start_time']].tolist()[0]
                overlapping_pairs.append((original_i, original_j, overlap_duration))
    
    return overlapping_pairs

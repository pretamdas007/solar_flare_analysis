"""
Solar Flare Model Fitting Module

This module implements the Gryciuk et al. flare model fitting to GOES XRS data.
The model uses a Gaussian convolved with an exponential decay to fit solar flare profiles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, differential_evolution
from scipy.special import erf
from scipy.signal import find_peaks
import json
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SolarFlareModel:
    """
    Implementation of the Gryciuk et al. solar flare model.
    
    The model function is:
    f(t) = (1/2) * sqrt(π) * A * C * exp[D(B-t) + (C²D²)/4] * [erf(Z) - erf((Z-t)/C)]
    where Z = (2B + C²D)/(2C)
    """    
    def __init__(self):
        self.parameters = ['A', 'B', 'C', 'D']
        # Updated bounds for real GOES XRS data
        # A: Amplitude bounds (real flares range from ~1e-9 to 1e-4 W/m²)
        # B: Peak time bounds (0 to 2 hours)
        # C: Width bounds (30 seconds to 1 hour)
        # D: Decay rate bounds (more realistic for actual flares)
        self.bounds = [(1e-10, 1e-4), (0, 7200), (30, 3600), (1e-5, 1e-2)]
    
    def model_function(self, t, A, B, C, D):
        """
        Gryciuk et al. flare model function
        
        Parameters:
        -----------
        t : array-like
            Time array
        A : float
            Amplitude (peak height)
        B : float
            Time of peak
        C : float
            Width/duration parameter
        D : float
            Decay rate
            
        Returns:
        --------
        array-like
            Model flux values
        """
        try:
            # Calculate Z parameter
            Z = (2 * B + C**2 * D) / (2 * C)
            
            # Calculate the exponential term
            exp_term = np.exp(D * (B - t) + (C**2 * D**2) / 4)
            
            # Calculate the error function terms
            erf_term1 = erf(Z)
            erf_term2 = erf((Z - t) / C)
            
            # Full model
            flux = 0.5 * np.sqrt(np.pi) * A * C * exp_term * (erf_term1 - erf_term2)
            
            return np.where(np.isfinite(flux), flux, 0)
        
        except (OverflowError, ZeroDivisionError, RuntimeWarning):
            return np.zeros_like(t)
    
    def objective_function(self, params, t, flux, weights=None):
        """
        Objective function for optimization (weighted least squares)
        """
        A, B, C, D = params
        model_flux = self.model_function(t, A, B, C, D)
        
        if weights is None:
            weights = np.ones_like(flux)
        
        residuals = weights * (flux - model_flux)
        return np.sum(residuals**2)
    
    def fit_flare(self, t, flux, method='differential_evolution', weights=None):
        """
        Fit the flare model to data
        
        Parameters:
        -----------
        t : array-like
            Time array
        flux : array-like
            Flux measurements
        method : str
            Optimization method ('differential_evolution' or 'minimize')
        weights : array-like, optional
            Weights for fitting
            
        Returns:
        --------
        dict
            Fitted parameters and fit quality metrics
        """
        # Normalize time to start from 0
        t_norm = t - t.min()
        
        # Initial guess based on data
        peak_idx = np.argmax(flux)
        A_guess = np.max(flux)
        B_guess = t_norm[peak_idx]
        C_guess = (t_norm.max() - t_norm.min()) / 4
        D_guess = 1e-4
        
        initial_guess = [A_guess, B_guess, C_guess, D_guess]
        
        try:
            if method == 'differential_evolution':
                result = differential_evolution(
                    self.objective_function,
                    bounds=self.bounds,
                    args=(t_norm, flux, weights),
                    seed=42,
                    maxiter=1000,
                    atol=1e-8,
                    tol=1e-8
                )
            else:
                result = minimize(
                    self.objective_function,
                    initial_guess,
                    args=(t_norm, flux, weights),
                    method='L-BFGS-B',
                    bounds=self.bounds
                )
            
            # Calculate fit quality metrics
            fitted_flux = self.model_function(t_norm, *result.x)
            r_squared = self.calculate_r_squared(flux, fitted_flux)
            rmse = np.sqrt(np.mean((flux - fitted_flux)**2))
            
            return {
                'success': result.success,
                'parameters': dict(zip(self.parameters, result.x)),
                'fitted_flux': fitted_flux,
                'r_squared': r_squared,
                'rmse': rmse,
                'original_time': t,
                'normalized_time': t_norm,
                'flux': flux
            }
            
        except Exception as e:
            print(f"Fitting failed: {e}")
            return {
                'success': False,
                'parameters': dict(zip(self.parameters, [np.nan]*4)),
                'fitted_flux': np.full_like(flux, np.nan),
                'r_squared': np.nan,
                'rmse': np.nan,
                'original_time': t,
                'normalized_time': t_norm,
                'flux': flux
            }
    
    def calculate_r_squared(self, observed, predicted):
        """Calculate R-squared statistic"""
        ss_res = np.sum((observed - predicted)**2)
        ss_tot = np.sum((observed - np.mean(observed))**2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

class FlareSegmentation:
    """
    Automatic flare detection and segmentation from GOES XRS data
    """
    
    def __init__(self, threshold_factor=2.0, min_duration=180):
        # Reduced threshold factor for real data (more sensitive)
        self.threshold_factor = threshold_factor
        # Reduced minimum duration for real flares (3 minutes)
        self.min_duration = min_duration  # minimum flare duration in seconds
    def detect_flares(self, time, flux, background_window=3600):
        """
        Detect flare events in GOES XRS flux data
        
        Parameters:
        -----------
        time : array-like
            Time array (seconds or datetime)
        flux : array-like
            Flux measurements
        background_window : int
            Window size for background estimation (seconds)
            
        Returns:
        --------
        list
            List of flare segments [(start_idx, end_idx), ...]
        """
        # Calculate rolling background
        background = self.calculate_rolling_background(flux, background_window)
        
        # Detect peaks above threshold
        threshold = background + self.threshold_factor * np.std(flux - background)
        peaks, _ = find_peaks(flux, height=threshold, distance=300)
        
        flare_segments = []
        
        for peak in peaks:
            # Find flare start and end
            start_idx = self.find_flare_start(flux, peak, background)
            end_idx = self.find_flare_end(flux, peak, background)
            
            # Check minimum duration
            # For GOES XRS data, time is already in seconds
            duration = time[end_idx] - time[start_idx]
            
            if duration >= self.min_duration:
                flare_segments.append((start_idx, end_idx))
        
        return flare_segments
    
    def calculate_rolling_background(self, flux, window):
        """Calculate rolling background level"""
        # For large datasets, use a more efficient approach
        # Use scipy for rolling statistics or pandas if available
        try:
            import pandas as pd
            # Convert to pandas for efficient rolling operations
            flux_series = pd.Series(flux)
            # Use rolling percentile for background estimation
            background = flux_series.rolling(window=window//60, min_periods=1, center=True).quantile(0.1)
            return background.values
        except ImportError:
            # Fallback to original method but with sampling for efficiency
            half_window = window // 2
            background = np.zeros_like(flux)
            
            # For large datasets, sample every 10th point to speed up computation
            step = max(1, len(flux) // 10000)  # Sample for efficiency on large datasets
            
            for i in range(0, len(flux), step):
                start = max(0, i - half_window)
                end = min(len(flux), i + half_window)
                bg_val = np.percentile(flux[start:end], 10)
                
                # Fill the gap
                end_fill = min(len(flux), i + step)
                background[i:end_fill] = bg_val
            
            # Fill any remaining gaps
            if step > 1:
                background = np.interp(np.arange(len(flux)), 
                                     np.arange(0, len(flux), step), 
                                     background[::step])
            
            return background
    
    def find_flare_start(self, flux, peak_idx, background):
        """Find flare start point"""
        threshold = background[peak_idx] + 0.5 * (flux[peak_idx] - background[peak_idx])
        
        for i in range(peak_idx, -1, -1):
            if flux[i] <= threshold:
                return i
        return 0
    
    def find_flare_end(self, flux, peak_idx, background):
        """Find flare end point"""
        threshold = background[peak_idx] + 0.5 * (flux[peak_idx] - background[peak_idx])
        
        for i in range(peak_idx, len(flux)):
            if flux[i] <= threshold:
                return i
        return len(flux) - 1

def load_goes_data(filepath):
    """
    Load GOES XRS data from CSV file
    
    Expected format: columns for time and flux
    Real GOES data format: time_minutes, time_seconds, xrsa_flux_observed, xrsb_flux_observed
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Loaded {len(data)} data points from {filepath}")
        print(f"Available columns: {list(data.columns)}")
        
        # Handle real GOES XRS data format
        if 'time_seconds' in data.columns and 'xrsa_flux_observed' in data.columns:
            # Use XRSA (1-8 Å) data as primary flux measurement
            # XRSA is typically used for flare classification
            time_col = 'time_seconds'
            flux_col = 'xrsa_flux_observed'
            
            # Filter out invalid/missing data
            valid_mask = (data[flux_col] > 0) & np.isfinite(data[flux_col]) & np.isfinite(data[time_col])
            data_clean = data[valid_mask].copy()
            
            print(f"Using {flux_col} as flux measurement")
            print(f"Data range: {data_clean[flux_col].min():.2e} to {data_clean[flux_col].max():.2e} W/m²")
            print(f"Valid data points: {len(data_clean)} / {len(data)}")
            
            return data_clean[[time_col, flux_col]].rename(columns={time_col: 'time', flux_col: 'flux'})
        
        # Try to identify time and flux columns for other formats
        time_cols = [col for col in data.columns if 'time' in col.lower()]
        flux_cols = [col for col in data.columns if any(x in col.lower() for x in ['flux', 'xrs', 'irradiance'])]
        
        if not time_cols or not flux_cols:
            print(f"Warning: Could not identify time/flux columns in {filepath}")
            print(f"Available columns: {list(data.columns)}")
            return None
        
        time_col = time_cols[0]
        flux_col = flux_cols[0]
        
        # Convert time to datetime if it's not already
        if data[time_col].dtype == 'object':
            data[time_col] = pd.to_datetime(data[time_col])
        
        # Filter out invalid data
        valid_mask = (data[flux_col] > 0) & np.isfinite(data[flux_col]) & np.isfinite(data[time_col])
        data_clean = data[valid_mask].copy()
        
        print(f"Valid data points: {len(data_clean)} / {len(data)}")
        
        return data_clean[[time_col, flux_col]].rename(columns={time_col: 'time', flux_col: 'flux'})
    
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def process_file(filepath, output_dir, plot=True):
    """
    Process a single GOES data file
    """
    print(f"Processing: {filepath}")
    
    # Load data
    data = load_goes_data(filepath)
    if data is None:
        return
    
    # Convert time to seconds from start
    if isinstance(data['time'].iloc[0], pd.Timestamp):
        time_seconds = (data['time'] - data['time'].iloc[0]).dt.total_seconds().values
    else:
        time_seconds = data['time'].values
    
    flux = data['flux'].values
    
    # Remove NaN values
    mask = np.isfinite(flux) & np.isfinite(time_seconds)
    time_seconds = time_seconds[mask]
    flux = flux[mask]
    
    if len(flux) < 100:
        print(f"Insufficient data points in {filepath}")
        return
    
    # Detect flares
    segmenter = FlareSegmentation()
    flare_segments = segmenter.detect_flares(time_seconds, flux)
    
    print(f"Found {len(flare_segments)} flare events")
    
    # Fit model to each flare
    model = SolarFlareModel()
    all_fits = []
    
    for i, (start, end) in enumerate(flare_segments):
        t_flare = time_seconds[start:end]
        flux_flare = flux[start:end]
        
        # Subtract background
        background = np.min(flux_flare)
        flux_flare = flux_flare - background
        
        fit_result = model.fit_flare(t_flare, flux_flare)
        fit_result['flare_id'] = i
        fit_result['file'] = os.path.basename(filepath)
        fit_result['background'] = background
        
        all_fits.append(fit_result)
        
        # Plot if requested
        if plot and fit_result['success']:
            plot_flare_fit(fit_result, output_dir, f"{os.path.basename(filepath)}_flare_{i}")
    
    # Save fitting results
    save_fits(all_fits, output_dir, os.path.basename(filepath))
    
    return all_fits

def plot_flare_fit(fit_result, output_dir, filename):
    """
    Plot flare fit results
    """
    plt.figure(figsize=(10, 6))
    
    t = fit_result['normalized_time']
    flux = fit_result['flux']
    fitted_flux = fit_result['fitted_flux']
    
    plt.plot(t, flux, 'ko-', label='Observed', markersize=3, alpha=0.7)
    plt.plot(t, fitted_flux, 'r-', label='Fitted Model', linewidth=2)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Flux (W/m²)')
    plt.title(f'Solar Flare Fit - R² = {fit_result["r_squared"]:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add parameter text
    params = fit_result['parameters']
    param_text = f"A = {params['A']:.2e}\nB = {params['B']:.1f}\nC = {params['C']:.1f}\nD = {params['D']:.2e}"
    plt.text(0.02, 0.98, param_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}_fit.png"), dpi=300, bbox_inches='tight')
    plt.close()

def save_fits(fits, output_dir, filename):
    """
    Save fitting results to JSON
    """
    # Convert numpy types to native Python types for JSON serialization
    fits_serializable = []
    for fit in fits:
        fit_copy = fit.copy()
        for key, value in fit_copy.items():
            if isinstance(value, np.ndarray):
                fit_copy[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                fit_copy[key] = value.item()
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, (np.integer, np.floating)):
                        fit_copy[key][k] = v.item()
        fits_serializable.append(fit_copy)
    
    output_file = os.path.join(output_dir, 'fits', f"{filename}_fits.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(fits_serializable, f, indent=2)

def main():
    """
    Main function to process GOES data files
    """
    parser = argparse.ArgumentParser(description='Fit solar flare models to GOES XRS data')
    parser.add_argument('--data_dir', default='data', help='Directory containing GOES CSV files')
    parser.add_argument('--output_dir', default='.', help='Output directory for results')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directories
    (output_dir / 'fits').mkdir(exist_ok=True)
    (output_dir / 'output').mkdir(exist_ok=True)
    
    # Find all CSV files
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        print("Please place GOES XRS data files in the data/ directory")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each file
    all_results = []
    for filepath in tqdm(csv_files):
        try:
            results = process_file(filepath, output_dir, plot=args.plot)
            if results:
                all_results.extend(results)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    print(f"\nProcessing complete. Successfully fitted {len(all_results)} flare events.")
    
    # Create summary statistics
    successful_fits = [r for r in all_results if r['success']]
    if successful_fits:
        r_squared_values = [r['r_squared'] for r in successful_fits]
        print(f"Mean R² = {np.mean(r_squared_values):.3f} ± {np.std(r_squared_values):.3f}")

if __name__ == '__main__':
    main()

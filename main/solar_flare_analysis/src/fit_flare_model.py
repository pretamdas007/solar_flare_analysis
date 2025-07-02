"""
Enhanced Solar Flare Model Fitting Module

This module implements the Gryciuk et al. flare model fitting to GOES XRS data
with enhanced smoothing, fitting methods, and professional seaborn visualizations.
The model uses a Gaussian convolved with an exponential decay to fit solar flare profiles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, differential_evolution, curve_fit
from scipy.special import erf
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
import json
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend and style
plt.style.use('default')
sns.set_palette("husl")

class EnhancedSolarFlareModel:
    """
    Enhanced implementation of the Gryciuk et al. solar flare model with smoothing
    and advanced fitting methods.
    
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
        Gryciuk et al. flare model function with robust error handling
        
        Parameters:
        -----------
        t : array-like
            Time array
        A : float
            Amplitude (peak height)
        B : float            Time of peak
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
            
            # Return finite values only
            return np.where(np.isfinite(flux), flux, 0)
            
        except Exception as e:
            # Return zeros if calculation fails
            return np.zeros_like(t)
    
    def smooth_data(self, flux, method='savgol', window_length=11, sigma=2):
        """
        Apply smoothing to flux data
        
        Parameters:
        -----------
        flux : array-like
            Input flux data
        method : str
            Smoothing method ('savgol' or 'gaussian')
        window_length : int
            Window length for Savitzky-Golay filter
        sigma : float
            Standard deviation for Gaussian filter
            
        Returns:
        --------
        array-like
            Smoothed flux data
        """
        if method == 'savgol' and len(flux) > window_length:
            # Ensure odd window length
            if window_length % 2 == 0:
                window_length += 1
            window_length = min(window_length, len(flux) - 1)
            if window_length < 3:                return flux
            try:
                return savgol_filter(flux, window_length, polyorder=2)
            except:
                return flux
        elif method == 'gaussian':
            return gaussian_filter1d(flux, sigma=sigma)
        else:
            return flux
    
    def objective_function(self, params, t, flux, weights=None):
        """
        Objective function for optimization (weighted least squares)
        
        Parameters:
        -----------
        params : array-like
            Model parameters [A, B, C, D]
        t : array-like
            Time array
        flux : array-like
            Observed flux
        weights : array-like, optional
            Weights for data points
            
        Returns:
        --------
        float
            Sum of squared residuals
        """
        A, B, C, D = params
        model_flux = self.model_function(t, A, B, C, D)
        
        if weights is None:
            weights = np.ones_like(flux)
        residuals = weights * (flux - model_flux)
        return np.sum(residuals**2)
    
    def fit_flare(self, t, flux, method='differential_evolution', weights=None, smooth=True):
        """
        Enhanced flare fitting with smoothing and multiple methods
        
        Parameters:
        -----------
        t : array-like
            Time array
        flux : array-like
            Flux measurements
        method : str
            Optimization method ('differential_evolution', 'minimize', or 'curve_fit')
        weights : array-like, optional
            Weights for fitting
        smooth : bool
            Whether to apply smoothing to data
            
        Returns:
        --------
        dict
            Fitted parameters and fit quality metrics
        """
        # Normalize time to start from 0
        t_norm = t - t.min()
        flux_original = flux.copy()
        
        # Apply smoothing if requested
        if smooth:
            flux = self.smooth_data(flux, method='savgol')
        
        # Initial guess based on data
        peak_idx = np.argmax(flux)
        A_guess = np.max(flux)
        B_guess = t_norm[peak_idx]
        C_guess = (t_norm.max() - t_norm.min()) / 4        
        D_guess = 1e-4
        
        initial_guess = [A_guess, B_guess, C_guess, D_guess]
        
        try:
            if method == 'curve_fit':
                popt, pcov = curve_fit(
                    self.model_function, t_norm, flux, p0=initial_guess,
                    bounds=([b[0] for b in self.bounds], [b[1] for b in self.bounds]),
                    maxfev=2000
                )
                result_x = popt
                success = True
            elif method == 'differential_evolution':
                result = differential_evolution(
                    self.objective_function,
                    bounds=self.bounds,
                    args=(t_norm, flux, weights),
                    seed=42,
                    maxiter=1000,
                    atol=1e-8,
                    tol=1e-8
                )
                result_x = result.x
                success = result.success
            else:  # minimize
                result = minimize(
                    self.objective_function,
                    initial_guess,
                    args=(t_norm, flux, weights),
                    method='L-BFGS-B',
                    bounds=self.bounds
                )
                result_x = result.x
                success = result.success
              # Calculate fit quality metrics
            fitted_flux = self.model_function(t_norm, *result_x)
            fitted_flux_original = self.model_function(t_norm, *result_x)
            r_squared = self.calculate_r_squared(flux_original, fitted_flux_original)
            rmse = np.sqrt(np.mean((flux_original - fitted_flux_original)**2))
            
            return {
                'success': success,
                'parameters': dict(zip(self.parameters, result_x)),
                'fitted_flux': fitted_flux,
                'fitted_flux_original': fitted_flux_original,
                'r_squared': r_squared,
                'rmse': rmse,
                'original_time': t,
                'normalized_time': t_norm,
                'flux': flux,
                'flux_original': flux_original,                
                'smoothed': smooth
            }
            
        except Exception as e:
            print(f"Fitting failed: {e}")
            return {
                'success': False,
                'parameters': dict(zip(self.parameters, [np.nan]*4)),
                'fitted_flux': np.full_like(flux, np.nan),
                'fitted_flux_original': np.full_like(flux_original, np.nan),
                'r_squared': np.nan,
                'rmse': np.nan,
                'original_time': t,
                'normalized_time': t_norm,
                'flux': flux,
                'flux_original': flux_original,
                'smoothed': smooth
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
    model = EnhancedSolarFlareModel()
    all_fits = []
    
    for i, (start, end) in enumerate(flare_segments):
        t_flare = time_seconds[start:end]
        flux_flare = flux[start:end]        
        # Subtract background
        background = np.min(flux_flare)
        flux_flare = flux_flare - background
        
        # Enhanced fitting with smoothing and multiple methods
        fit_result = model.fit_flare(t_flare, flux_flare, method='differential_evolution', smooth=True)
        fit_result['flare_id'] = i
        fit_result['file'] = os.path.basename(filepath)
        fit_result['background'] = background
        fit_result['smoothed'] = True  # Mark that smoothing was applied
        
        all_fits.append(fit_result)
        
        # Plot if requested (now uses enhanced 4-panel visualization)
        if plot and fit_result['success']:
            plot_flare_fit(fit_result, output_dir, f"{os.path.basename(filepath)}_flare_{i}")
    
    # Save fitting results
    save_fits(all_fits, output_dir, os.path.basename(filepath))
    
    return all_fits

def plot_enhanced_flare_fit(fit_result, output_dir, filename):
    """
    Create enhanced 4-panel seaborn visualization of flare fit
    """
    # Set up the figure with seaborn style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Enhanced Solar Flare Model Analysis with Seaborn', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    t = fit_result['normalized_time']
    flux = fit_result['flux']
    fitted_flux = fit_result['fitted_flux']
    params = fit_result['parameters']
    
    # Plot 1: Data and Fit with enhanced styling
    axes[0, 0].plot(t, flux, 'o', color='steelblue', markersize=4, alpha=0.7, 
                   label='Observed Data', markeredgecolor='darkblue', markeredgewidth=0.5)
    axes[0, 0].plot(t, fitted_flux, '-', color='crimson', linewidth=3, 
                   label='Gryciuk Model Fit', alpha=0.9)
    
    axes[0, 0].set_xlabel('Time (seconds)', fontweight='bold')
    axes[0, 0].set_ylabel('Flux (W/m²)', fontweight='bold')
    axes[0, 0].set_title('Data vs Model Fit', fontweight='bold', pad=15)
    axes[0, 0].legend(loc='upper right', framealpha=0.9)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=axes[0, 0])
    
    # Plot 2: Residuals with enhanced styling
    residuals = flux - fitted_flux
    axes[0, 1].plot(t, residuals, 'o', color='darkgreen', markersize=3, alpha=0.7,
                   markeredgecolor='forestgreen', markeredgewidth=0.5)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    axes[0, 1].set_xlabel('Time (seconds)', fontweight='bold')
    axes[0, 1].set_ylabel('Residuals (W/m²)', fontweight='bold')
    axes[0, 1].set_title('Fit Residuals', fontweight='bold', pad=15)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=axes[0, 1])
    
    # Plot 3: Parameter values as bar chart
    param_names = ['A', 'B', 'C', 'D']
    param_values = [params[p] for p in param_names]
    
    bars = axes[1, 0].bar(param_names, param_values, color='steelblue', alpha=0.8,
                         edgecolor='darkblue', linewidth=1)
    axes[1, 0].set_ylabel('Parameter Value', fontweight='bold')
    axes[1, 0].set_title('Fitted Parameters', fontweight='bold', pad=15)
    sns.despine(ax=axes[1, 0])
    
    # Add value labels on bars
    for bar, val in zip(bars, param_values):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                       f'{val:.2e}', ha='center', va='bottom', fontweight='bold',
                       fontsize=9)
    
    # Plot 4: Summary information
    axes[1, 1].axis('off')
    
    info_text = "ENHANCED FIT ANALYSIS\\n" + "="*25 + "\\n\\n"
    info_text += f"📊 Model Performance:\\n"
    info_text += f"   • R²: {fit_result['r_squared']:.4f}\\n"
    info_text += f"   • RMSE: {fit_result['rmse']:.2e}\\n\\n"
    
    info_text += f"🔧 Parameters:\\n"
    for param, value in params.items():
        info_text += f"   • {param}: {value:.3e}\\n"
    
    info_text += f"\\n⚙️ Processing:\\n"
    info_text += f"   • Data points: {len(t)}\\n"
    info_text += f"   • Duration: {t.max():.1f}s\\n"
    info_text += f"   • Smoothed: {'Yes' if fit_result.get('smoothed') else 'No'}\\n"
    
    # Color-code based on R²
    r_sq = fit_result['r_squared']
    bg_color = 'lightgreen' if r_sq > 0.8 else 'lightyellow' if r_sq > 0.5 else 'lightcoral'
    
    axes[1, 1].text(0.05, 0.95, info_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.6', facecolor=bg_color, alpha=0.7))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the enhanced plot
    output_path = os.path.join(output_dir, 'fits', f"{filename}_enhanced_fit.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_flare_fit(fit_result, output_dir, filename):
    """
    Compatibility wrapper - now uses enhanced plotting
    """
    plot_enhanced_flare_fit(fit_result, output_dir, filename)

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
    Main function to process GOES data files with enhanced features
    """
    parser = argparse.ArgumentParser(description='Fit solar flare models to GOES XRS data with enhanced features')
    parser.add_argument('--data_dir', default='data', help='Directory containing GOES CSV files')
    parser.add_argument('--output_dir', default='.', help='Output directory for results')
    parser.add_argument('--plot', action='store_true', help='Generate enhanced plots')
    parser.add_argument('--smooth', action='store_true', default=True, help='Apply data smoothing')
    parser.add_argument('--smooth_method', default='savgol', choices=['savgol', 'gaussian', 'none'],
                       help='Smoothing method to use')
    parser.add_argument('--fit_method', default='auto', 
                       choices=['auto', 'curve_fit', 'differential_evolution', 'minimize'],
                       help='Fitting method to use')
    
    args = parser.parse_args()
    
    # Try to find data directory automatically if default doesn't exist
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        # Try different possible locations
        possible_data_dirs = [
            Path('data'),
            Path('../data'), 
            Path('./data'),
            Path('main/solar_flare_analysis/data'),
            Path('../main/solar_flare_analysis/data')
        ]
        
        for possible_dir in possible_data_dirs:
            if possible_dir.exists() and list(possible_dir.glob('*.csv')):
                data_dir = possible_dir
                print(f"Found data directory: {data_dir.absolute()}")
                break
    
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
    print(f"Enhanced features: smoothing={args.smooth}, smooth_method={args.smooth_method}")
    print(f"Fitting method: {args.fit_method}")
    
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

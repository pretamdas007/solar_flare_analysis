"""
Enhanced Solar Flare Model Fitting Module

This module implements the Gryciuk et al. flare model fitting to GOES XRS data
with enhanced smoothing, fitting methods, and seaborn visualizations.
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

class EnhancedSolarFlareModel:
    """
    Enhanced implementation of the Gryciuk et al. solar flare model with smoothing
    """
    
    def __init__(self):
        self.parameters = ['A', 'B', 'C', 'D']
        self.bounds = [(1e-10, 1e-4), (0, 7200), (30, 3600), (1e-5, 1e-2)]
    
    def model_function(self, t, A, B, C, D):
        """Gryciuk et al. flare model function"""
        try:
            Z = (2 * B + C**2 * D) / (2 * C)
            exp_term = np.exp(D * (B - t) + (C**2 * D**2) / 4)
            erf_term1 = erf(Z)
            erf_term2 = erf((Z - t) / C)
            flux = 0.5 * np.sqrt(np.pi) * A * C * exp_term * (erf_term1 - erf_term2)
            return np.where(np.isfinite(flux), flux, 0)
        except:
            return np.zeros_like(t)
    
    def smooth_data(self, flux, method='savgol', window_length=11, sigma=2):
        """Apply smoothing to flux data"""
        if method == 'savgol' and len(flux) > window_length:
            if window_length % 2 == 0:
                window_length += 1
            window_length = min(window_length, len(flux) - 1)
            if window_length < 3:
                return flux
            try:
                return savgol_filter(flux, window_length, polyorder=2)
            except:
                return flux
        elif method == 'gaussian':
            return gaussian_filter1d(flux, sigma=sigma)
        else:
            return flux
    
    def objective_function(self, params, t, flux, weights=None):
        """Objective function for optimization"""
        A, B, C, D = params
        model_flux = self.model_function(t, A, B, C, D)
        if weights is None:
            weights = np.ones_like(flux)
        residuals = weights * (flux - model_flux)
        return np.sum(residuals**2)
    
    def fit_flare(self, t, flux, method='differential_evolution', smooth=True):
        """Enhanced flare fitting with smoothing"""
        t_norm = t - t.min()
        flux_original = flux.copy()
        
        if smooth:
            flux = self.smooth_data(flux, method='savgol')
        
        # Initial guess
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
            else:
                result = differential_evolution(
                    self.objective_function,
                    bounds=self.bounds,
                    args=(t_norm, flux, None),
                    seed=42,
                    maxiter=1000
                )
                result_x = result.x
                success = result.success
            
            # Calculate metrics
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

def plot_enhanced_flare_fit(fit_result, output_dir, filename):
    """Enhanced seaborn visualization of flare fit"""
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 11
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    t = fit_result['normalized_time']
    flux = fit_result['flux']
    flux_original = fit_result.get('flux_original', flux)
    fitted_flux_original = fit_result.get('fitted_flux_original', fit_result['fitted_flux'])
    
    # Plot 1: Main fit
    axes[0, 0].plot(t, flux_original, 'o', color='steelblue', markersize=4, alpha=0.7, 
                   label='Original Data', markeredgecolor='darkblue', markeredgewidth=0.5)
    
    if fit_result.get('smoothed', False):
        axes[0, 0].plot(t, flux, '-', color='orange', linewidth=2, alpha=0.8, 
                       label='Smoothed Data')
    
    axes[0, 0].plot(t, fitted_flux_original, '-', color='crimson', linewidth=3, 
                   label='Fitted Model', alpha=0.9)
    
    axes[0, 0].set_xlabel('Time (seconds)', fontweight='bold')
    axes[0, 0].set_ylabel('Flux (W/m²)', fontweight='bold')
    axes[0, 0].set_title(f'Enhanced Solar Flare Fit - R² = {fit_result["r_squared"]:.3f}', 
                        fontweight='bold', pad=15)
    axes[0, 0].legend(frameon=True, fancybox=True, shadow=True)
    sns.despine(ax=axes[0, 0])
    
    # Plot 2: Residuals
    residuals = flux_original - fitted_flux_original
    axes[0, 1].scatter(fitted_flux_original, residuals, alpha=0.7, color='steelblue')
    axes[0, 1].axhline(y=0, color='crimson', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Fitted Values', fontweight='bold')
    axes[0, 1].set_ylabel('Residuals', fontweight='bold')
    axes[0, 1].set_title('Residuals Analysis', fontweight='bold', pad=15)
    sns.despine(ax=axes[0, 1])
    
    # Plot 3: Parameters
    params = fit_result['parameters']
    param_names = list(params.keys())
    param_values = list(params.values())
    
    bars = axes[1, 0].bar(param_names, param_values, color='steelblue', alpha=0.8,
                         edgecolor='darkblue', linewidth=1)
    axes[1, 0].set_ylabel('Parameter Value', fontweight='bold')
    axes[1, 0].set_title('Fitted Parameters', fontweight='bold', pad=15)
    sns.despine(ax=axes[1, 0])
    
    # Add value labels
    for bar, val in zip(bars, param_values):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                       f'{val:.2e}', ha='center', va='bottom', fontweight='bold',
                       fontsize=9)
    
    # Plot 4: Summary
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
    
    r_sq = fit_result['r_squared']
    bg_color = 'lightgreen' if r_sq > 0.8 else 'lightyellow' if r_sq > 0.5 else 'lightcoral'
    
    axes[1, 1].text(0.05, 0.95, info_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.6', facecolor=bg_color, alpha=0.7))
    
    plt.suptitle('Enhanced Solar Flare Model Analysis with Seaborn', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = os.path.join(output_dir, 'fits', f"{filename}_enhanced_fit.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# Self-contained data loading functions
def load_goes_data(filepath):
    """Load GOES XRS data from CSV file"""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} data points from {filepath}")
        print(f"Available columns: {list(df.columns)}")
        
        # Determine time and flux columns
        time_col = None
        flux_col = None
        
        if 'time_seconds' in df.columns:
            time_col = 'time_seconds'
        elif 'time_minutes' in df.columns:
            time_col = 'time_minutes'
            df['time_seconds'] = df[time_col] * 60
        
        # Use XRSA flux by preference
        if 'xrsa_flux_observed' in df.columns:
            flux_col = 'xrsa_flux_observed'
        elif 'xrsb_flux_observed' in df.columns:
            flux_col = 'xrsb_flux_observed'
        elif 'flux' in df.columns:
            flux_col = 'flux'
        
        if time_col is None or flux_col is None:
            print(f"Could not identify time/flux columns")
            return None
        
        print(f"Using {flux_col} as flux measurement")
        
        # Clean data
        df = df.dropna(subset=['time_seconds', flux_col])
        flux_vals = df[flux_col].values
        time_vals = df['time_seconds'].values
        
        # Remove invalid values
        valid_mask = (flux_vals > 0) & np.isfinite(flux_vals) & np.isfinite(time_vals)
        flux_vals = flux_vals[valid_mask]
        time_vals = time_vals[valid_mask]
        
        print(f"Data range: {np.min(flux_vals):.2e} to {np.max(flux_vals):.2e} W/m²")
        print(f"Valid data points: {len(flux_vals)} / {len(df)}")
        
        return pd.DataFrame({'time': time_vals, 'flux': flux_vals})
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def detect_flares_simple(time_vals, flux_vals, threshold_factor=3.0):
    """Simple flare detection using peak finding"""
    from scipy.signal import find_peaks
    
    # Calculate rolling baseline
    window_size = min(100, len(flux_vals) // 10)
    if window_size < 5:
        window_size = 5
    
    baseline = np.convolve(flux_vals, np.ones(window_size)/window_size, mode='same')
    
    # Find peaks above threshold
    peak_threshold = baseline + threshold_factor * np.std(flux_vals)
    peaks, _ = find_peaks(flux_vals, height=peak_threshold, distance=50)
    
    # Create segments around peaks
    segments = []
    for peak in peaks:
        start = max(0, peak - 100)
        end = min(len(flux_vals), peak + 200)
        if end - start > 50:  # Minimum segment length
            segments.append((start, end))
    
    return segments

# Simple test with one file
def test_enhanced_fitting():
    """Test the enhanced fitting on a sample data file"""
    # Load one data file for testing - check multiple possible locations
    possible_data_dirs = [
        Path('data'),                    # Current directory
        Path('../data'),                 # One level up
        Path('./data'),                  # Explicit current
        Path('main/solar_flare_analysis/data'),  # From project root
        Path('../main/solar_flare_analysis/data')  # From parent
    ]
    
    data_files = []
    for data_dir in possible_data_dirs:
        if data_dir.exists():
            files = list(data_dir.glob('*.csv'))
            if files:
                data_files = files
                print(f"Found {len(data_files)} CSV files in {data_dir}")
                break
    
    if not data_files:
        print("No data files found in any of the expected locations:")
        for data_dir in possible_data_dirs:
            print(f"  - {data_dir.absolute()} (exists: {data_dir.exists()})")
        return
    
    print(f"Testing enhanced fitting on {data_files[0]}")
    data = load_goes_data(data_files[0])
    
    if data is None:
        print("Failed to load data")
        return
    
    # Get first 50000 points for testing
    time_seconds = data['time'].values[:50000]
    flux = data['flux'].values[:50000]
    
    # Detect flares
    flare_segments = detect_flares_simple(time_seconds, flux)
    
    print(f"Found {len(flare_segments)} flare events in test data")
    
    if len(flare_segments) > 0:
        # Test on first flare
        start, end = flare_segments[0]
        t_flare = time_seconds[start:end]
        flux_flare = flux[start:end]
        
        # Subtract baseline
        flux_flare = flux_flare - np.min(flux_flare)
        
        # Test enhanced model
        model = EnhancedSolarFlareModel()
        fit_result = model.fit_flare(t_flare, flux_flare, smooth=True)
        
        print(f"Fit successful: {fit_result['success']}")
        print(f"R²: {fit_result['r_squared']:.3f}")
        print(f"RMSE: {fit_result['rmse']:.2e}")
        
        # Create enhanced plot
        plot_enhanced_flare_fit(fit_result, '.', 'test_enhanced_flare')
        print("Enhanced plot saved as fits/test_enhanced_flare_enhanced_fit.png")

if __name__ == '__main__':
    test_enhanced_fitting()

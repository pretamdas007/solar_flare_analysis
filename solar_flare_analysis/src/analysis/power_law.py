"""
Analysis module for flare energy statistics and power-law distributions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import powerlaw
from astropy.modeling import models, fitting
from astropy.stats import bootstrap
from astropy.utils import NumpyRNGContext


def calculate_flare_energy(flare_data, flux_column, background_column=None, time_column=None):
    """
    Calculate the energy released during solar flares.
    
    Parameters
    ----------
    flare_data : pandas.DataFrame
        DataFrame containing flare information
    flux_column : str
        Name of the column containing flux measurements
    background_column : str, optional
        Name of the column containing background flux
    time_column : str, optional
        Name of the column containing time values
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with added energy calculations
    """
    # Create a copy to avoid modifying the original
    df = flare_data.copy()
    
    # If background column is not specified, use zeros
    if background_column is None:
        df['background'] = 0
    else:
        df['background'] = df[background_column]
    
    # Calculate background-subtracted flux
    df['flux_bgsub'] = df[flux_column] - df['background']
    
    # Set negative values to zero
    df['flux_bgsub'] = df['flux_bgsub'].clip(lower=0)
    
    # If time column is provided, calculate energy using trapezoidal integration
    if time_column is not None:
        # Make sure time is sorted
        df = df.sort_values(time_column)
        
        # Calculate time differences in seconds
        if pd.api.types.is_datetime64_any_dtype(df[time_column]):
            time_diff = df[time_column].diff().dt.total_seconds()
        else:
            time_diff = df[time_column].diff()
        
        # Handle NaN in the first position
        time_diff = time_diff.fillna(0)
        
        # Calculate energy using trapezoidal rule: E = ∫ flux(t) dt
        df['energy_increment'] = df['flux_bgsub'] * time_diff
        df['energy'] = df['energy_increment'].cumsum()
    
    return df


def fit_power_law(data, xmin=None, xmax=None, n_bootstrap=1000, plot=False):
    """
    Fit a power-law distribution to data and estimate uncertainties using bootstrap.
    
    Parameters
    ----------
    data : array-like
        Data to fit (e.g., flare energies)
    xmin : float, optional
        Minimum value for fitting
    xmax : float, optional
        Maximum value for fitting
    n_bootstrap : int, optional
        Number of bootstrap samples for uncertainty estimation
    plot : bool, optional
        If True, plot the fit results
        
    Returns
    -------
    dict
        Dictionary containing fit results and uncertainties    """
    # Filter out non-positive values (required for power-law fitting)
    data = np.array(data)
    data = data[data > 0]
    
    # Check if we have enough data points
    if len(data) == 0:
        print("Warning: No positive data points available for power-law fitting")
        return {
            'alpha': np.nan,
            'alpha_err': np.nan,
            'xmin': np.nan,
            'n_data': 0,
            'fit_results': None
        }
    
    if len(data) < 10:
        print(f"Warning: Only {len(data)} data points available. Power-law fitting may be unreliable.")
        if len(data) < 3:
            return {
                'alpha': np.nan,
                'alpha_err': np.nan,
                'xmin': np.nan,
                'n_data': len(data),
                'fit_results': None
            }
    
    # Initial power-law fit
    results = powerlaw.Fit(data, xmin=xmin, xmax=xmax)
    alpha = results.alpha
    xmin_fit = results.xmin if xmin is None else xmin
      # Bootstrap for uncertainty estimation (only if we have enough data)
    alpha_bootstraps = []
    if len(data) >= 10 and n_bootstrap > 0:
        with NumpyRNGContext(42):  # For reproducibility
            bootstrapped_samples = bootstrap(data, n_bootstrap)
        
        for bootstrap_sample in bootstrapped_samples:
            try:
                boot_fit = powerlaw.Fit(bootstrap_sample, xmin=xmin_fit, xmax=xmax)
                alpha_bootstraps.append(boot_fit.alpha)
            except:
                # Skip failed fits
                pass
      # Calculate uncertainty from bootstrap results
    alpha_err = np.std(alpha_bootstraps) if alpha_bootstraps else np.nan
    
    # Compare with other distributions (only if we have enough data)
    try:
        if len(data) >= 10:
            R, p = results.distribution_compare('power_law', 'lognormal')
        else:
            R, p = np.nan, np.nan
    except:
        R, p = np.nan, np.nan
      # If requested, plot the results
    if plot and len(data) >= 3:
        plt.figure(figsize=(12, 8))
        
        # Plot data and fit
        plt.subplot(2, 1, 1)
        results.plot_pdf(color='b', label='Data')
        results.power_law.plot_pdf(color='r', linestyle='--', label='Power-law Fit')
        plt.title(f'Power-law Fit: α = {alpha:.3f} ± {alpha_err:.3f if not np.isnan(alpha_err) else "N/A"}')
        plt.legend()
        plt.grid(True)
        
        # Plot bootstrap distribution of alpha (only if we have bootstrap results)
        if alpha_bootstraps:
            plt.subplot(2, 1, 2)
            plt.hist(alpha_bootstraps, bins=30, alpha=0.7)
            plt.axvline(alpha, color='r', linestyle='--', 
                       label=f'α = {alpha:.3f} ± {alpha_err:.3f}')
            plt.title('Bootstrap Distribution of Power-law Exponent')
            plt.xlabel('α')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
        else:
            plt.subplot(2, 1, 2)
            plt.text(0.5, 0.5, 'Insufficient data for bootstrap analysis', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Bootstrap Distribution (Not Available)')
        
        plt.tight_layout()
        plt.show()
    elif plot:
        print(f"Insufficient data for plotting (only {len(data)} points)")
    
    # Return the results
    return {
        'alpha': alpha,
        'alpha_err': alpha_err,
        'xmin': xmin_fit,
        'xmax': xmax,
        'n_data': len(data),
        'R_lognormal': R,
        'p_lognormal': p,
        'bootstrap_samples': alpha_bootstraps
    }


def compare_flare_populations(energies1, labels1, energies2, labels2, 
                              xmin=None, xmax=None, plot=False):
    """
    Compare power-law fits between two flare populations.
    
    Parameters
    ----------
    energies1 : array-like
        Energies for the first population
    labels1 : str
        Label for the first population
    energies2 : array-like
        Energies for the second population
    labels2 : str
        Label for the second population
    xmin : float, optional
        Minimum value for fitting
    xmax : float, optional
        Maximum value for fitting
    plot : bool, optional
        If True, plot the comparison
        
    Returns
    -------
    dict
        Dictionary containing comparison results
    """
    # Fit power laws to both populations
    fit1 = fit_power_law(energies1, xmin=xmin, xmax=xmax, plot=False)
    fit2 = fit_power_law(energies2, xmin=xmin, xmax=xmax, plot=False)
    
    # Calculate statistical significance of the difference
    alpha_diff = abs(fit1['alpha'] - fit2['alpha'])
    alpha_err_combined = np.sqrt(fit1['alpha_err']**2 + fit2['alpha_err']**2)
    significance = alpha_diff / alpha_err_combined
    
    # If requested, plot the comparison
    if plot:
        plt.figure(figsize=(12, 8))
        
        # Plot both distributions and fits
        plt.subplot(2, 1, 1)
        
        # Plot data using powerlaw's plotting functions
        pl1 = powerlaw.Fit(energies1, xmin=fit1['xmin'], xmax=xmax)
        pl2 = powerlaw.Fit(energies2, xmin=fit2['xmin'], xmax=xmax)
        
        pl1.plot_pdf(color='b', label=f'{labels1} Data')
        pl1.power_law.plot_pdf(color='b', linestyle='--', 
                              label=f'{labels1} Fit: α = {fit1["alpha"]:.3f} ± {fit1["alpha_err"]:.3f}')
        
        pl2.plot_pdf(color='r', label=f'{labels2} Data')
        pl2.power_law.plot_pdf(color='r', linestyle='--',
                              label=f'{labels2} Fit: α = {fit2["alpha"]:.3f} ± {fit2["alpha_err"]:.3f}')
        
        plt.title(f'Power-law Comparison (Significance: {significance:.2f}σ)')
        plt.legend()
        plt.grid(True)
        
        # Plot bootstrap distributions
        plt.subplot(2, 1, 2)
        plt.hist(fit1['bootstrap_samples'], bins=30, alpha=0.5, color='b', label=labels1)
        plt.hist(fit2['bootstrap_samples'], bins=30, alpha=0.5, color='r', label=labels2)
        plt.axvline(fit1['alpha'], color='b', linestyle='--')
        plt.axvline(fit2['alpha'], color='r', linestyle='--')
        plt.title('Bootstrap Distributions of Power-law Exponents')
        plt.xlabel('α')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    # Return the comparison results
    return {
        'fit1': fit1,
        'fit2': fit2,
        'alpha_diff': alpha_diff,
        'alpha_err_combined': alpha_err_combined,
        'significance': significance,
        'p_value': 2 * (1 - stats.norm.cdf(significance))  # Two-tailed p-value
    }


def flare_frequency_distribution(energies, energy_bins=20, fit_range=None, plot=False):
    """
    Calculate and optionally plot the flare frequency distribution.
    
    Parameters
    ----------
    energies : array-like
        Flare energies
    energy_bins : int or array-like, optional
        Number of bins or bin edges for energy histogram
    fit_range : tuple, optional
        (min, max) energy range for fitting
    plot : bool, optional
        If True, plot the distribution
        
    Returns
    -------
    tuple
        Histogram data (bin centers, counts) and fit results
    """
    # Calculate histogram in log space
    log_energies = np.log10(np.array(energies)[np.array(energies) > 0])
    
    if isinstance(energy_bins, int):
        # Generate logarithmic bins
        min_energy = np.min(log_energies)
        max_energy = np.max(log_energies)
        bins = np.logspace(min_energy, max_energy, energy_bins)
    else:
        bins = energy_bins
    
    # Calculate histogram
    hist, bin_edges = np.histogram(energies, bins=bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    # Calculate frequency (counts per energy bin)
    energy_bin_widths = np.diff(bin_edges)
    frequency = hist / energy_bin_widths
    
    # Fit power-law in log-log space
    if fit_range is not None:
        min_e, max_e = fit_range
        mask = (bin_centers >= min_e) & (bin_centers <= max_e)
        x_fit = bin_centers[mask]
        y_fit = frequency[mask]
    else:
        # Exclude empty bins
        mask = hist > 0
        x_fit = bin_centers[mask]
        y_fit = frequency[mask]
    
    # Log-transformed fit
    log_x = np.log10(x_fit)
    log_y = np.log10(y_fit)
    
    # Linear fit in log-log space
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
    
    # If requested, plot the distribution
    if plot:
        plt.figure(figsize=(10, 6))
        
        # Plot frequency distribution
        plt.loglog(bin_centers, frequency, 'bo', label='Data')
        
        # Plot power-law fit
        x_plot = np.logspace(np.log10(min(x_fit)), np.log10(max(x_fit)), 100)
        y_plot = 10**intercept * x_plot**slope
        plt.loglog(x_plot, y_plot, 'r-', 
                  label=f'Power Law: α = {-slope:.3f} ± {std_err:.3f}')
        
        plt.title('Flare Frequency Distribution')
        plt.xlabel('Energy')
        plt.ylabel('Frequency (counts/bin width)')
        plt.legend()
        plt.grid(True, which='both', ls='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    # Return histogram data and fit results
    return (bin_centers, frequency), {
        'slope': slope,
        'intercept': intercept,
        'alpha': -slope,  # Power-law exponent
        'alpha_err': std_err,
        'r_value': r_value,
        'p_value': p_value
    }

"""
Enhanced analysis module for flare energy statistics and power-law distributions.

This module provides comprehensive tools for analyzing solar flare energy distributions,
including robust power-law fitting, statistical testing, and comparative analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle, Ellipse
import seaborn as sns
from scipy import stats, optimize, integrate
from scipy.interpolate import interp1d
import powerlaw
import warnings
from typing import Optional, Union, Tuple, Dict, List, Any
from dataclasses import dataclass
from pathlib import Path
import logging

# Optional imports for enhanced visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("Plotly not available. Interactive plots will be disabled.")

try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False

# Optional imports with fallbacks
try:
    from astropy.modeling import models, fitting
    from astropy.stats import bootstrap
    from astropy.utils import NumpyRNGContext
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
    warnings.warn("Astropy not available. Some advanced features will be disabled.")

try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Professional styling configuration
plt.style.use('default')  # Start with clean slate
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# Custom color palettes
FLARE_PALETTE = {
    'primary': '#E74C3C',      # Vibrant red
    'secondary': '#3498DB',    # Clear blue  
    'tertiary': '#2ECC71',     # Emerald green
    'quaternary': '#9B59B6',   # Purple
    'accent': '#F39C12',       # Orange
    'dark': '#34495E',         # Dark blue-gray
    'light': '#ECF0F1',        # Light gray
    'background': '#FFFFFF',   # White
    'grid': '#BDC3C7',         # Light gray
    'text': '#2C3E50'          # Dark gray
}

DISTRIBUTION_PALETTE = {
    'power_law': '#E74C3C',
    'lognormal': '#3498DB',
    'exponential': '#2ECC71',
    'power_law_cutoff': '#9B59B6',
    'stretched_exponential': '#F39C12',
    'weibull': '#E67E22'
}

# Professional matplotlib configuration
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#2C3E50',
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    'grid.color': '#BDC3C7',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 2.5,
    'lines.markersize': 6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
})


@dataclass
class PowerLawResults:
    """Container for power-law fitting results."""
    alpha: float
    alpha_err: float
    xmin: float
    xmax: Optional[float]
    n_data: int
    ks_statistic: float
    p_value: float
    log_likelihood: float
    aic: float
    bic: float
    bootstrap_samples: List[float]
    distribution_comparison: Dict[str, Any]
    goodness_of_fit: Dict[str, float]


@dataclass
class FlareStatistics:
    """Container for comprehensive flare statistics."""
    n_flares: int
    total_energy: float
    mean_energy: float
    median_energy: float
    energy_std: float
    energy_range: Tuple[float, float]
    duration_stats: Dict[str, float]
    peak_flux_stats: Dict[str, float]
    temporal_distribution: Dict[str, Any]


def calculate_flare_energy(
    flare_data: pd.DataFrame, 
    flux_column: str, 
    background_column: Optional[str] = None, 
    time_column: Optional[str] = None,
    integration_method: str = 'trapz',
    energy_units: str = 'J',
    flux_conversion_factor: float = 1.0
) -> pd.DataFrame:
    """
    Calculate the energy released during solar flares with enhanced options.
    
    Parameters
    ----------
    flare_data : pd.DataFrame
        DataFrame containing flare information
    flux_column : str
        Name of the column containing flux measurements
    background_column : str, optional
        Name of the column containing background flux
    time_column : str, optional
        Name of the column containing time values
    integration_method : str, default 'trapz'
        Integration method: 'trapz', 'simpson', or 'cumulative'
    energy_units : str, default 'J'
        Units for energy calculation
    flux_conversion_factor : float, default 1.0
        Factor to convert flux units if needed
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added energy calculations and statistics
    """
    # Create a copy to avoid modifying the original
    df = flare_data.copy()
    
    # Apply flux conversion if needed
    df[flux_column] = df[flux_column] * flux_conversion_factor
    
    # Handle background subtraction
    if background_column is None:
        df['background'] = 0
        logger.info("No background column specified, using zero background")
    else:
        df['background'] = df[background_column] * flux_conversion_factor
    
    # Calculate background-subtracted flux
    df['flux_bgsub'] = df[flux_column] - df['background']
    
    # Set negative values to zero and log warnings
    negative_count = (df['flux_bgsub'] < 0).sum()
    if negative_count > 0:
        logger.warning(f"Found {negative_count} negative flux values, setting to zero")
        df['flux_bgsub'] = df['flux_bgsub'].clip(lower=0)
    
    # Calculate additional flux statistics
    df['flux_snr'] = np.where(df['background'] > 0, 
                             df['flux_bgsub'] / df['background'], 
                             np.inf)
    
    # If time column is provided, calculate energy using specified integration method
    if time_column is not None:
        # Ensure data is sorted by time
        df = df.sort_values(time_column).reset_index(drop=True)
        
        # Calculate time differences in seconds
        if pd.api.types.is_datetime64_any_dtype(df[time_column]):
            time_values = df[time_column].astype('datetime64[ns]')
            time_seconds = (time_values - time_values.iloc[0]).dt.total_seconds().values
        else:
            time_seconds = df[time_column].values
        
        # Calculate energy using different integration methods
        if integration_method == 'trapz':
            # Trapezoidal integration
            energy_cumulative = np.zeros_like(df['flux_bgsub'])
            for i in range(1, len(df)):
                energy_increment = 0.5 * (df['flux_bgsub'].iloc[i] + df['flux_bgsub'].iloc[i-1]) * \
                                 (time_seconds[i] - time_seconds[i-1])
                energy_cumulative[i] = energy_cumulative[i-1] + energy_increment
            df['energy'] = energy_cumulative
            
        elif integration_method == 'simpson' and len(df) >= 3:
            # Simpson's rule integration
            try:
                from scipy.integrate import cumtrapz, simpson
                df['energy'] = cumtrapz(df['flux_bgsub'], time_seconds, initial=0)
            except ImportError:
                logger.warning("SciPy not available for Simpson's rule, falling back to trapezoidal")
                df['energy'] = np.cumsum(df['flux_bgsub'] * np.gradient(time_seconds))
                
        else:  # cumulative method
            time_diff = np.gradient(time_seconds)
            df['energy_increment'] = df['flux_bgsub'] * time_diff
            df['energy'] = df['energy_increment'].cumsum()
    
    else:
        # If no time column, assume unit time steps
        logger.info("No time column provided, using unit time steps")
        df['energy'] = df['flux_bgsub'].cumsum()
    
    # Add energy statistics
    df['log_energy'] = np.log10(df['energy'].replace(0, np.nan))
    df['energy_normalized'] = df['energy'] / df['energy'].max() if df['energy'].max() > 0 else 0
    
    # Add metadata
    df.attrs = {
        'energy_units': energy_units,
        'integration_method': integration_method,
        'flux_conversion_factor': flux_conversion_factor,
        'total_energy': df['energy'].iloc[-1] if len(df) > 0 else 0,
        'peak_flux': df[flux_column].max(),
        'background_mean': df['background'].mean()
    }
    
    return df


def fit_power_law(
    data: Union[np.ndarray, List[float]], 
    xmin: Optional[float] = None, 
    xmax: Optional[float] = None, 
    n_bootstrap: int = 1000,
    bootstrap_method: str = 'astropy',
    plot: bool = False,
    save_plot: Optional[str] = None,
    fit_method: str = 'mle',
    distribution_tests: List[str] = ['lognormal', 'exponential', 'power_law_cutoff'],
    confidence_level: float = 0.95
) -> PowerLawResults:
    """
    Enhanced power-law fitting with comprehensive statistical analysis.
    
    Parameters
    ----------
    data : array-like
        Data to fit (e.g., flare energies)
    xmin : float, optional
        Minimum value for fitting. If None, automatically determined
    xmax : float, optional
        Maximum value for fitting
    n_bootstrap : int, default 1000
        Number of bootstrap samples for uncertainty estimation
    bootstrap_method : str, default 'astropy'
        Bootstrap method: 'astropy', 'manual', or 'parametric'
    plot : bool, default False
        If True, create comprehensive plots
    save_plot : str, optional
        Path to save the plot
    fit_method : str, default 'mle'
        Fitting method: 'mle' (maximum likelihood) or 'ks' (Kolmogorov-Smirnov)
    distribution_tests : list, default ['lognormal', 'exponential', 'power_law_cutoff']
        Alternative distributions to compare against
    confidence_level : float, default 0.95
        Confidence level for uncertainty estimates
        
    Returns
    -------
    PowerLawResults
        Comprehensive results object with all fitting statistics
    """
    # Input validation and preprocessing
    data = np.array(data, dtype=float)
    data = data[np.isfinite(data) & (data > 0)]
    
    if len(data) == 0:
        logger.error("No positive, finite data points available for power-law fitting")
        return _empty_power_law_results()
    
    if len(data) < 10:
        logger.warning(f"Only {len(data)} data points available. Results may be unreliable.")
        if len(data) < 3:
            return _empty_power_law_results()
    
    # Determine optimal xmin if not provided
    if xmin is None:
        xmin = _find_optimal_xmin(data)
        logger.info(f"Automatically determined xmin = {xmin:.2e}")
    
    # Filter data based on xmin/xmax
    data_filtered = data[data >= xmin]
    if xmax is not None:
        data_filtered = data_filtered[data_filtered <= xmax]
    
    if len(data_filtered) < 3:
        logger.error(f"Insufficient data points ({len(data_filtered)}) after filtering")
        return _empty_power_law_results()
    
    # Primary power-law fit
    try:
        if fit_method == 'mle':
            results = powerlaw.Fit(data_filtered, xmin=xmin, xmax=xmax)
        else:  # KS method
            results = powerlaw.Fit(data_filtered, xmin=xmin, xmax=xmax, parameter_range={'alpha': [1.0, 10.0]})
        
        alpha = results.alpha
        xmin_fit = results.xmin
        ks_stat = results.D
        
    except Exception as e:
        logger.error(f"Power-law fitting failed: {e}")
        return _empty_power_law_results()
    
    # Bootstrap uncertainty estimation
    alpha_bootstraps = []
    if n_bootstrap > 0 and len(data_filtered) >= 10:
        alpha_bootstraps = _bootstrap_alpha(data_filtered, xmin_fit, xmax, n_bootstrap, bootstrap_method)
    
    # Calculate uncertainties
    alpha_err = np.std(alpha_bootstraps) if alpha_bootstraps else np.nan
    alpha_ci = np.percentile(alpha_bootstraps, [(1-confidence_level)/2*100, (1+confidence_level)/2*100]) \
               if alpha_bootstraps else [np.nan, np.nan]
    
    # Distribution comparisons
    distribution_comparison = {}
    for dist in distribution_tests:
        try:
            R, p = results.distribution_compare('power_law', dist, normalized_ratio=True)
            distribution_comparison[dist] = {'R': R, 'p': p}
        except Exception as e:
            logger.warning(f"Distribution comparison with {dist} failed: {e}")
            distribution_comparison[dist] = {'R': np.nan, 'p': np.nan}
    
    # Goodness-of-fit tests
    goodness_of_fit = _goodness_of_fit_tests(data_filtered, alpha, xmin_fit, xmax)
    
    # Calculate information criteria
    log_likelihood = results.loglikelihoods[0] if hasattr(results, 'loglikelihoods') else np.nan
    n_params = 1  # alpha is the only parameter
    aic = 2 * n_params - 2 * log_likelihood if not np.isnan(log_likelihood) else np.nan
    bic = np.log(len(data_filtered)) * n_params - 2 * log_likelihood if not np.isnan(log_likelihood) else np.nan
    
    # Create results object
    power_law_results = PowerLawResults(
        alpha=alpha,
        alpha_err=alpha_err,
        xmin=xmin_fit,
        xmax=xmax,
        n_data=len(data_filtered),
        ks_statistic=ks_stat,
        p_value=goodness_of_fit.get('p_value', np.nan),
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        bootstrap_samples=alpha_bootstraps,
        distribution_comparison=distribution_comparison,
        goodness_of_fit=goodness_of_fit
    )
    
    # Create plots if requested
    if plot:
        _plot_power_law_analysis(data_filtered, power_law_results, results, save_plot)
    
    return power_law_results


def _empty_power_law_results() -> PowerLawResults:
    """Return empty results object for failed fits."""
    return PowerLawResults(
        alpha=np.nan, alpha_err=np.nan, xmin=np.nan, xmax=None, n_data=0,
        ks_statistic=np.nan, p_value=np.nan, log_likelihood=np.nan,
        aic=np.nan, bic=np.nan, bootstrap_samples=[], 
        distribution_comparison={}, goodness_of_fit={}
    )


def _find_optimal_xmin(data: np.ndarray) -> float:
    """Find optimal xmin using Kolmogorov-Smirnov statistic."""
    try:
        # Try multiple candidate xmin values
        data_sorted = np.sort(data)
        n_candidates = min(50, len(data_sorted) // 10)
        candidates = data_sorted[::len(data_sorted)//n_candidates][:n_candidates]
        
        best_xmin = candidates[0]
        best_ks = np.inf
        
        for xmin_candidate in candidates:
            try:
                fit = powerlaw.Fit(data, xmin=xmin_candidate)
                if fit.D < best_ks:
                    best_ks = fit.D
                    best_xmin = xmin_candidate
            except:
                continue
        
        return best_xmin
    except:
        return np.percentile(data, 10)  # Fallback to 10th percentile


def _bootstrap_alpha(data: np.ndarray, xmin: float, xmax: Optional[float], 
                     n_bootstrap: int, method: str) -> List[float]:
    """Perform bootstrap resampling to estimate alpha uncertainty."""
    alpha_samples = []
    
    if method == 'astropy' and HAS_ASTROPY:
        try:
            with NumpyRNGContext(42):
                bootstrapped_samples = bootstrap(data, n_bootstrap)
            
            for bootstrap_sample in bootstrapped_samples:
                try:
                    boot_fit = powerlaw.Fit(bootstrap_sample, xmin=xmin, xmax=xmax)
                    alpha_samples.append(boot_fit.alpha)
                except:
                    continue
        except Exception as e:
            logger.warning(f"Astropy bootstrap failed: {e}, falling back to manual method")
            method = 'manual'
    
    if method == 'manual' or not HAS_ASTROPY:
        np.random.seed(42)
        for _ in range(n_bootstrap):
            try:
                bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                boot_fit = powerlaw.Fit(bootstrap_sample, xmin=xmin, xmax=xmax)
                alpha_samples.append(boot_fit.alpha)
            except:
                continue
    
    elif method == 'parametric':
        # Parametric bootstrap: generate synthetic data from fitted distribution
        original_fit = powerlaw.Fit(data, xmin=xmin, xmax=xmax)
        alpha_orig = original_fit.alpha
        
        for _ in range(n_bootstrap):
            try:
                # Generate synthetic power-law data
                synthetic_data = original_fit.power_law.generate_random(len(data))
                boot_fit = powerlaw.Fit(synthetic_data, xmin=xmin, xmax=xmax)
                alpha_samples.append(boot_fit.alpha)
            except:
                continue
    
    return alpha_samples


def _goodness_of_fit_tests(data: np.ndarray, alpha: float, xmin: float, 
                          xmax: Optional[float]) -> Dict[str, float]:
    """Perform comprehensive goodness-of-fit tests."""
    tests = {}
    
    try:
        # Kolmogorov-Smirnov test
        fit = powerlaw.Fit(data, xmin=xmin, xmax=xmax)
        tests['ks_statistic'] = fit.D
        
        # Anderson-Darling test (if available)
        try:
            from scipy.stats import anderson
            # Convert to empirical CDF for AD test
            data_sorted = np.sort(data)
            empirical_cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
            theoretical_cdf = 1 - (data_sorted / xmin) ** (1 - alpha)
            ad_stat, _, ad_p = anderson(empirical_cdf - theoretical_cdf)
            tests['anderson_darling'] = ad_stat
            tests['p_value'] = ad_p
        except:
            tests['p_value'] = np.nan
        
        # Cramér-von Mises test
        try:
            data_normalized = data / xmin
            n = len(data_normalized)
            expected_cdf = 1 - data_normalized ** (1 - alpha)
            empirical_cdf = np.arange(1, n + 1) / n
            cvm_stat = np.sum((empirical_cdf - expected_cdf) ** 2)
            tests['cramer_von_mises'] = cvm_stat
        except:
            tests['cramer_von_mises'] = np.nan
            
    except Exception as e:
        logger.warning(f"Goodness-of-fit tests failed: {e}")
        tests = {'ks_statistic': np.nan, 'p_value': np.nan}
    
    return tests


def compare_flare_populations(
    energies1: Union[np.ndarray, List[float]], 
    labels1: str, 
    energies2: Union[np.ndarray, List[float]], 
    labels2: str,
    xmin: Optional[float] = None, 
    xmax: Optional[float] = None, 
    plot: bool = False,
    save_plot: Optional[str] = None,
    statistical_tests: List[str] = ['ks', 'anderson', 'bootstrap'],
    n_bootstrap: int = 1000
) -> Dict[str, Any]:
    """
    Enhanced comparison of power-law fits between two flare populations.
    
    Parameters
    ----------
    energies1, energies2 : array-like
        Energies for the two populations
    labels1, labels2 : str
        Labels for the populations
    xmin, xmax : float, optional
        Range for fitting
    plot : bool, default False
        Whether to create comparison plots
    save_plot : str, optional
        Path to save plots
    statistical_tests : list, default ['ks', 'anderson', 'bootstrap']
        Statistical tests to perform
    n_bootstrap : int, default 1000
        Number of bootstrap samples for significance testing
        
    Returns
    -------
    dict
        Comprehensive comparison results
    """
    logger.info(f"Comparing flare populations: {labels1} vs {labels2}")
    
    # Convert to numpy arrays and filter
    energies1 = np.array(energies1)
    energies2 = np.array(energies2)
    energies1 = energies1[energies1 > 0]
    energies2 = energies2[energies2 > 0]
    
    if len(energies1) == 0 or len(energies2) == 0:
        logger.error("One or both populations have no valid data")
        return {'error': 'Insufficient data for comparison'}
    
    # Fit power laws to both populations
    fit1 = fit_power_law(energies1, xmin=xmin, xmax=xmax, plot=False)
    fit2 = fit_power_law(energies2, xmin=xmin, xmax=xmax, plot=False)
    
    # Calculate basic comparison statistics
    alpha_diff = abs(fit1.alpha - fit2.alpha)
    alpha_err_combined = np.sqrt(fit1.alpha_err**2 + fit2.alpha_err**2)
    significance = alpha_diff / alpha_err_combined if alpha_err_combined > 0 else np.nan
    
    comparison_results = {
        'fit1': fit1,
        'fit2': fit2,
        'alpha_difference': alpha_diff,
        'alpha_error_combined': alpha_err_combined,
        'significance_sigma': significance,
        'p_value_gaussian': 2 * (1 - stats.norm.cdf(significance)) if not np.isnan(significance) else np.nan
    }
    
    # Statistical tests
    test_results = {}
    
    if 'ks' in statistical_tests:
        # Kolmogorov-Smirnov test
        try:
            ks_stat, ks_p = stats.ks_2samp(energies1, energies2)
            test_results['ks_test'] = {'statistic': ks_stat, 'p_value': ks_p}
        except Exception as e:
            logger.warning(f"KS test failed: {e}")
            test_results['ks_test'] = {'statistic': np.nan, 'p_value': np.nan}
    
    if 'anderson' in statistical_tests:
        # Anderson-Darling test
        try:
            ad_stat, ad_critical, ad_significance = stats.anderson_ksamp([energies1, energies2])
            test_results['anderson_test'] = {
                'statistic': ad_stat,
                'critical_values': ad_critical,
                'significance_level': ad_significance
            }
        except Exception as e:
            logger.warning(f"Anderson-Darling test failed: {e}")
            test_results['anderson_test'] = {'statistic': np.nan}
    
    if 'bootstrap' in statistical_tests and len(fit1.bootstrap_samples) > 0 and len(fit2.bootstrap_samples) > 0:
        # Bootstrap significance test
        try:
            bootstrap_diff = []
            min_samples = min(len(fit1.bootstrap_samples), len(fit2.bootstrap_samples))
            for i in range(min_samples):
                diff = abs(fit1.bootstrap_samples[i] - fit2.bootstrap_samples[i])
                bootstrap_diff.append(diff)
            
            test_results['bootstrap_test'] = {
                'mean_difference': np.mean(bootstrap_diff),
                'std_difference': np.std(bootstrap_diff),
                'percentiles': np.percentile(bootstrap_diff, [5, 25, 50, 75, 95])
            }
        except Exception as e:
            logger.warning(f"Bootstrap test failed: {e}")
    
    comparison_results['statistical_tests'] = test_results
    
    # Energy distribution comparison
    energy_comparison = _compare_energy_distributions(energies1, energies2, labels1, labels2)
    comparison_results['energy_comparison'] = energy_comparison
    
    # Model selection criteria
    model_selection = _model_selection_comparison(fit1, fit2, labels1, labels2)
    comparison_results['model_selection'] = model_selection
    
    # Create plots if requested
    if plot:
        _plot_population_comparison(energies1, energies2, fit1, fit2, 
                                  labels1, labels2, comparison_results, save_plot)
    
    return comparison_results


def _compare_energy_distributions(energies1: np.ndarray, energies2: np.ndarray,
                                labels1: str, labels2: str) -> Dict[str, Any]:
    """Compare energy distributions between populations."""
    comparison = {}
    
    try:
        # Basic statistics comparison
        stats1 = {
            'mean': np.mean(energies1),
            'median': np.median(energies1),
            'std': np.std(energies1),
            'min': np.min(energies1),
            'max': np.max(energies1),
            'n': len(energies1)
        }
        
        stats2 = {
            'mean': np.mean(energies2),
            'median': np.median(energies2),
            'std': np.std(energies2),
            'min': np.min(energies2),
            'max': np.max(energies2),
            'n': len(energies2)
        }
        
        comparison['basic_stats'] = {labels1: stats1, labels2: stats2}
        
        # Ratio comparisons
        comparison['ratios'] = {
            'mean_ratio': stats1['mean'] / stats2['mean'] if stats2['mean'] > 0 else np.inf,
            'median_ratio': stats1['median'] / stats2['median'] if stats2['median'] > 0 else np.inf,
            'std_ratio': stats1['std'] / stats2['std'] if stats2['std'] > 0 else np.inf,
            'range_overlap': _calculate_range_overlap(
                (stats1['min'], stats1['max']), 
                (stats2['min'], stats2['max'])
            )
        }
        
        # Percentile comparison
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        perc1 = np.percentile(energies1, percentiles)
        perc2 = np.percentile(energies2, percentiles)
        
        comparison['percentile_comparison'] = {
            'percentiles': percentiles,
            labels1: perc1.tolist(),
            labels2: perc2.tolist(),
            'ratios': (perc1 / perc2).tolist()
        }
        
    except Exception as e:
        logger.warning(f"Energy distribution comparison failed: {e}")
    
    return comparison


def _calculate_range_overlap(range1: Tuple[float, float], 
                           range2: Tuple[float, float]) -> float:
    """Calculate the overlap between two ranges."""
    min1, max1 = range1
    min2, max2 = range2
    
    overlap_min = max(min1, min2)
    overlap_max = min(max1, max2)
    
    if overlap_max <= overlap_min:
        return 0.0
    
    overlap_size = overlap_max - overlap_min
    total_size = max(max1, max2) - min(min1, min2)
    
    return overlap_size / total_size if total_size > 0 else 0.0


def _model_selection_comparison(fit1: PowerLawResults, fit2: PowerLawResults,
                              labels1: str, labels2: str) -> Dict[str, Any]:
    """Compare models using information criteria."""
    comparison = {}
    
    try:
        # AIC comparison
        if not np.isnan(fit1.aic) and not np.isnan(fit2.aic):
            aic_diff = fit1.aic - fit2.aic
            comparison['aic'] = {
                labels1: fit1.aic,
                labels2: fit2.aic,
                'difference': aic_diff,
                'preferred': labels1 if aic_diff < 0 else labels2,
                'strength': _interpret_aic_difference(abs(aic_diff))
            }
        
        # BIC comparison
        if not np.isnan(fit1.bic) and not np.isnan(fit2.bic):
            bic_diff = fit1.bic - fit2.bic
            comparison['bic'] = {
                labels1: fit1.bic,
                labels2: fit2.bic,
                'difference': bic_diff,
                'preferred': labels1 if bic_diff < 0 else labels2,
                'strength': _interpret_bic_difference(abs(bic_diff))
            }
        
        # Goodness-of-fit comparison
        comparison['goodness_of_fit'] = {
            labels1: {
                'ks_statistic': fit1.ks_statistic,
                'p_value': fit1.p_value,
                'n_data': fit1.n_data
            },
            labels2: {
                'ks_statistic': fit2.ks_statistic,
                'p_value': fit2.p_value,
                'n_data': fit2.n_data
            }
        }
        
    except Exception as e:
        logger.warning(f"Model selection comparison failed: {e}")
    
    return comparison


def _interpret_aic_difference(delta_aic: float) -> str:
    """Interpret AIC difference."""
    if delta_aic < 2:
        return "Weak evidence"
    elif delta_aic < 4:
        return "Positive evidence"
    elif delta_aic < 7:
        return "Strong evidence"
    else:
        return "Very strong evidence"


def _interpret_bic_difference(delta_bic: float) -> str:
    """Interpret BIC difference."""
    if delta_bic < 2:
        return "Weak evidence"
    elif delta_bic < 6:
        return "Positive evidence"
    elif delta_bic < 10:
        return "Strong evidence"
    else:
        return "Very strong evidence"


def _plot_population_comparison(energies1: np.ndarray, energies2: np.ndarray,
                              fit1: PowerLawResults, fit2: PowerLawResults,
                              labels1: str, labels2: str,
                              comparison_results: Dict[str, Any],
                              save_path: Optional[str] = None):
    """Create comprehensive comparison plots with enhanced styling."""
    # Set up professional styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20, 16))
    
    # Create sophisticated grid layout
    gs = gridspec.GridSpec(3, 4, height_ratios=[2, 1.5, 1], width_ratios=[2, 1.5, 1.5, 1],
                          hspace=0.4, wspace=0.35)
    
    # Enhanced color scheme
    color1 = FLARE_PALETTE['primary']
    color2 = FLARE_PALETTE['secondary']
    accent_color = FLARE_PALETTE['accent']
    
    # 1. Main comparison plot with advanced styling
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Calculate bins for both populations
    all_energies = np.concatenate([energies1, energies2])
    bins = np.logspace(np.log10(all_energies.min()), np.log10(all_energies.max()), 60)
    
    # Plot histograms with enhanced styling
    n1, bins1, patches1 = ax1.hist(energies1, bins=bins, alpha=0.6, density=True, 
                                  label=f'{labels1} (n={len(energies1):,})', 
                                  color=color1, edgecolor='white', linewidth=0.8)
    
    n2, bins2, patches2 = ax1.hist(energies2, bins=bins, alpha=0.6, density=True, 
                                  label=f'{labels2} (n={len(energies2):,})', 
                                  color=color2, edgecolor='white', linewidth=0.8)
    
    # Add gradient effects to histograms
    for i, (p1, p2) in enumerate(zip(patches1, patches2)):
        alpha_val = 0.3 + 0.7 * (i / len(patches1))
        p1.set_alpha(alpha_val)
        p2.set_alpha(alpha_val)
    
    # Plot power-law fits with confidence bands
    x_range = np.logspace(np.log10(max(fit1.xmin, fit2.xmin)), 
                         np.log10(min(np.max(energies1), np.max(energies2))), 200)
    
    # Fit 1
    y1 = (fit1.alpha - 1) / fit1.xmin * (x_range / fit1.xmin) ** (-fit1.alpha)
    ax1.plot(x_range, y1, '--', color=color1, linewidth=3,
            label=f'{labels1} Fit: α = {fit1.alpha:.3f} ± {fit1.alpha_err:.3f}')
    
    # Fit 2
    y2 = (fit2.alpha - 1) / fit2.xmin * (x_range / fit2.xmin) ** (-fit2.alpha)
    ax1.plot(x_range, y2, '--', color=color2, linewidth=3,
            label=f'{labels2} Fit: α = {fit2.alpha:.3f} ± {fit2.alpha_err:.3f}')
    
    # Add confidence bands if available
    if fit1.bootstrap_samples and len(fit1.bootstrap_samples) > 10:
        alpha1_lower, alpha1_upper = np.percentile(fit1.bootstrap_samples, [2.5, 97.5])
        y1_lower = (alpha1_lower - 1) / fit1.xmin * (x_range / fit1.xmin) ** (-alpha1_lower)
        y1_upper = (alpha1_upper - 1) / fit1.xmin * (x_range / fit1.xmin) ** (-alpha1_upper)
        ax1.fill_between(x_range, y1_lower, y1_upper, alpha=0.2, color=color1)
    
    if fit2.bootstrap_samples and len(fit2.bootstrap_samples) > 10:
        alpha2_lower, alpha2_upper = np.percentile(fit2.bootstrap_samples, [2.5, 97.5])
        y2_lower = (alpha2_lower - 1) / fit2.xmin * (x_range / fit2.xmin) ** (-alpha2_lower)
        y2_upper = (alpha2_upper - 1) / fit2.xmin * (x_range / fit2.xmin) ** (-alpha2_upper)
        ax1.fill_between(x_range, y2_lower, y2_upper, alpha=0.2, color=color2)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Energy', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=14, fontweight='bold')
    
    # Enhanced title with statistical significance
    significance = comparison_results.get('significance_sigma', 0)
    title_text = f'Population Comparison: {labels1} vs {labels2}\n'
    title_text += f'Δα = {comparison_results["alpha_difference"]:.4f} '
    title_text += f'({significance:.2f}σ significance)'
    if significance > 2:
        title_text += ' [Significant Difference]'
    elif significance > 1:
        title_text += ' [Moderate Difference]'
    else:
        title_text += ' [Weak Difference]'
    
    ax1.set_title(title_text, fontsize=16, fontweight='bold', pad=20)
    
    # Enhanced legend
    legend = ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
                       borderpad=1, fontsize=11)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    ax1.grid(True, alpha=0.3)
    
    # 2. Bootstrap comparison with violin plots
    ax2 = fig.add_subplot(gs[0, 2])
    if fit1.bootstrap_samples and fit2.bootstrap_samples:
        # Create violin plot
        violin_data = [fit1.bootstrap_samples, fit2.bootstrap_samples]
        parts = ax2.violinplot(violin_data, positions=[1, 2], widths=0.6, 
                              showmeans=True, showmedians=True)
        
        # Color the violins
        colors_violin = [color1, color2]
        for pc, color in zip(parts['bodies'], colors_violin):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('white')
            pc.set_linewidth(1)
        
        # Add individual points with jitter
        for i, (data, color) in enumerate(zip(violin_data, colors_violin)):
            if len(data) < 1000:  # Only show points if reasonable number
                y = np.random.normal(i+1, 0.04, size=len(data))
                ax2.scatter(y, data, alpha=0.3, s=8, color=color, edgecolors='white', linewidth=0.3)
        
        # Add mean lines
        ax2.hlines(fit1.alpha, 0.7, 1.3, colors=color1, linestyles='--', linewidth=3, alpha=0.8)
        ax2.hlines(fit2.alpha, 1.7, 2.3, colors=color2, linestyles='--', linewidth=3, alpha=0.8)
        
        ax2.set_xticks([1, 2])
        ax2.set_xticklabels([labels1, labels2], fontweight='bold')
        ax2.set_ylabel('Power-law Exponent (α)', fontweight='bold')
        ax2.set_title('Bootstrap Distributions\nComparison', fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3)
        
        # Add statistical annotation
        diff_mean = np.mean(fit1.bootstrap_samples) - np.mean(fit2.bootstrap_samples)
        diff_std = np.sqrt(np.var(fit1.bootstrap_samples) + np.var(fit2.bootstrap_samples))
        ax2.text(0.5, 0.95, f'Δα = {diff_mean:.4f}\nσ = {diff_std:.4f}', 
                transform=ax2.transAxes, ha='center', va='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=accent_color, alpha=0.3),
                fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Bootstrap Data\nNot Available', 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=FLARE_PALETTE['light'], alpha=0.8))
    
    # 3. Statistical tests heatmap
    ax3 = fig.add_subplot(gs[0, 3])
    if 'statistical_tests' in comparison_results:
        tests = comparison_results['statistical_tests']
        
        # Create a summary of test results
        test_names = []
        test_values = []
        test_colors = []
        
        if 'ks_test' in tests:
            test_names.append('KS Test\np-value')
            p_val = tests['ks_test']['p_value']
            test_values.append(f'{p_val:.4f}')
            # Color based on significance
            if p_val < 0.001:
                test_colors.append('red')
            elif p_val < 0.01:
                test_colors.append('orange')
            elif p_val < 0.05:
                test_colors.append('yellow')
            else:
                test_colors.append('green')
        
        if 'anderson_test' in tests:
            test_names.append('Anderson-\nDarling')
            ad_stat = tests['anderson_test']['statistic']
            test_values.append(f'{ad_stat:.3f}')
            test_colors.append('lightblue')
        
        # Gaussian significance
        test_names.append('Gaussian\nSignificance')
        gauss_sig = comparison_results.get('significance_sigma', 0)
        test_values.append(f'{gauss_sig:.2f}σ')
        if gauss_sig > 3:
            test_colors.append('red')
        elif gauss_sig > 2:
            test_colors.append('orange')
        elif gauss_sig > 1:
            test_colors.append('yellow')
        else:
            test_colors.append('lightgreen')
        
        # Create colored boxes
        for i, (name, value, color) in enumerate(zip(test_names, test_values, test_colors)):
            rect = Rectangle((0, i), 1, 1, facecolor=color, alpha=0.7, edgecolor='white', linewidth=2)
            ax3.add_patch(rect)
            ax3.text(0.5, i + 0.7, name, ha='center', va='center', fontweight='bold', fontsize=9)
            ax3.text(0.5, i + 0.3, value, ha='center', va='center', fontweight='bold', fontsize=11)
        
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, len(test_names))
        ax3.set_title('Statistical Tests\nSummary', fontweight='bold', pad=15)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['left'].set_visible(False)
    
    # 4. Cumulative distribution comparison
    ax4 = fig.add_subplot(gs[1, 0])
    x1_sorted = np.sort(energies1)
    x2_sorted = np.sort(energies2)
    y1 = 1 - np.arange(1, len(x1_sorted) + 1) / len(x1_sorted)
    y2 = 1 - np.arange(1, len(x2_sorted) + 1) / len(x2_sorted)
    
    ax4.loglog(x1_sorted, y1, color=color1, linewidth=3, alpha=0.8, label=labels1)
    ax4.loglog(x2_sorted, y2, color=color2, linewidth=3, alpha=0.8, label=labels2)
    
    # Add theoretical lines
    x_theory = np.logspace(np.log10(max(fit1.xmin, fit2.xmin)), 
                          np.log10(min(np.max(energies1), np.max(energies2))), 100)
    y1_theory = (x_theory / fit1.xmin) ** (1 - fit1.alpha)
    y2_theory = (x_theory / fit2.xmin) ** (1 - fit2.alpha)
    
    ax4.loglog(x_theory, y1_theory, '--', color=color1, alpha=0.6, linewidth=2)
    ax4.loglog(x_theory, y2_theory, '--', color=color2, alpha=0.6, linewidth=2)
    
    ax4.set_xlabel('Energy', fontweight='bold')
    ax4.set_ylabel('P(X ≥ x)', fontweight='bold')
    ax4.set_title('Complementary CDF\nComparison', fontweight='bold', pad=15)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. Energy statistics comparison
    ax5 = fig.add_subplot(gs[1, 1])
    if 'energy_comparison' in comparison_results:
        ec = comparison_results['energy_comparison']
        
        if 'basic_stats' in ec:
            stats1 = ec['basic_stats'][labels1]
            stats2 = ec['basic_stats'][labels2]
            
            # Create comparison bars
            metrics = ['Mean', 'Median', 'Std Dev']
            values1 = [stats1['mean'], stats1['median'], stats1['std']]
            values2 = [stats2['mean'], stats2['median'], stats2['std']]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            bars1 = ax5.bar(x - width/2, values1, width, label=labels1, color=color1, alpha=0.8)
            bars2 = ax5.bar(x + width/2, values2, width, label=labels2, color=color2, alpha=0.8)
            
            # Add value labels on bars
            for bars, values in [(bars1, values1), (bars2, values2)]:
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{value:.2e}', ha='center', va='bottom', fontsize=8, rotation=45)
            
            ax5.set_yscale('log')
            ax5.set_xlabel('Statistical Metrics', fontweight='bold')
            ax5.set_ylabel('Energy Value', fontweight='bold')
            ax5.set_title('Energy Statistics\nComparison', fontweight='bold', pad=15)
            ax5.set_xticks(x)
            ax5.set_xticklabels(metrics)
            ax5.legend(fontsize=9)
            ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Model selection summary
    ax6 = fig.add_subplot(gs[1, 2])
    if 'model_selection' in comparison_results:
        ms = comparison_results['model_selection']
        
        # Create model selection visualization
        criteria = []
        preferences = []
        strengths = []
        
        if 'aic' in ms:
            criteria.append('AIC')
            preferences.append(ms['aic']['preferred'])
            strengths.append(ms['aic']['strength'])
        
        if 'bic' in ms:
            criteria.append('BIC')
            preferences.append(ms['bic']['preferred'])
            strengths.append(ms['bic']['strength'])
        
        # Create preference chart
        colors_pref = [color1 if pref == labels1 else color2 for pref in preferences]
        strength_values = [4 if 'Very strong' in s else 3 if 'Strong' in s else 2 if 'Positive' in s else 1 
                          for s in strengths]
        
        bars = ax6.barh(criteria, strength_values, color=colors_pref, alpha=0.8)
        
        # Add strength labels
        for i, (bar, strength) in enumerate(zip(bars, strengths)):
            ax6.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    strength, ha='left', va='center', fontweight='bold', fontsize=9)
        
        ax6.set_xlabel('Evidence Strength', fontweight='bold')
        ax6.set_title('Model Selection\nCriteria', fontweight='bold', pad=15)
        ax6.set_xlim(0, 5)
        ax6.grid(True, alpha=0.3, axis='x')
    
    # 7. Detailed comparison summary
    ax7 = fig.add_subplot(gs[1, 3])
    summary_text = []
    summary_text.append("COMPARISON SUMMARY")
    summary_text.append("=" * 20)
    summary_text.append(f"Population 1: {labels1}")
    summary_text.append(f"  α = {fit1.alpha:.3f} ± {fit1.alpha_err:.3f}")
    summary_text.append(f"  n = {fit1.n_data:,}")
    summary_text.append("")
    summary_text.append(f"Population 2: {labels2}")
    summary_text.append(f"  α = {fit2.alpha:.3f} ± {fit2.alpha_err:.3f}")
    summary_text.append(f"  n = {fit2.n_data:,}")
    summary_text.append("")
    summary_text.append("DIFFERENCE ANALYSIS")
    summary_text.append("-" * 18)
    summary_text.append(f"Δα = {comparison_results['alpha_difference']:.4f}")
    summary_text.append(f"Significance: {comparison_results.get('significance_sigma', 0):.2f}σ")
    
    if 'statistical_tests' in comparison_results and 'ks_test' in comparison_results['statistical_tests']:
        ks_p = comparison_results['statistical_tests']['ks_test']['p_value']
        summary_text.append(f"KS p-value: {ks_p:.4f}")
    
    # Interpretation
    sig_level = comparison_results.get('significance_sigma', 0)
    if sig_level > 3:
        interpretation = "HIGHLY SIGNIFICANT"
        box_color = 'red'
    elif sig_level > 2:
        interpretation = "SIGNIFICANT"
        box_color = 'orange'
    elif sig_level > 1:
        interpretation = "MODERATELY SIGNIFICANT"
        box_color = 'yellow'
    else:
        interpretation = "NOT SIGNIFICANT"
        box_color = 'lightgreen'
    
    summary_text.append("")
    summary_text.append(f"INTERPRETATION:")
    summary_text.append(interpretation)
    
    ax7.text(0.05, 0.95, '\n'.join(summary_text), transform=ax7.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.8", facecolor=box_color, alpha=0.3,
                     edgecolor='black', linewidth=1))
    ax7.axis('off')
    
    # 8. Bottom summary panel
    ax8 = fig.add_subplot(gs[2, :])
    
    # Create comprehensive summary
    summary_details = []
    
    # Basic comparison
    alpha_diff = comparison_results['alpha_difference']
    significance = comparison_results.get('significance_sigma', 0)
    
    summary_details.append(f"POWER-LAW COMPARISON ANALYSIS")
    summary_details.append(f"Populations: {labels1} (α={fit1.alpha:.3f}±{fit1.alpha_err:.3f}, n={fit1.n_data:,}) vs "
                         f"{labels2} (α={fit2.alpha:.3f}±{fit2.alpha_err:.3f}, n={fit2.n_data:,})")
    summary_details.append(f"Difference: Δα = {alpha_diff:.4f} ({significance:.2f}σ significance)")
    
    # Add statistical test results
    if 'statistical_tests' in comparison_results:
        tests = comparison_results['statistical_tests']
        test_summary = []
        if 'ks_test' in tests:
            ks = tests['ks_test']
            test_summary.append(f"KS: D={ks['statistic']:.4f}, p={ks['p_value']:.4f}")
        if 'anderson_test' in tests:
            ad = tests['anderson_test']
            test_summary.append(f"AD: {ad['statistic']:.4f}")
        
        if test_summary:
            summary_details.append(f"Statistical Tests: {' | '.join(test_summary)}")
    
    # Add model selection results
    if 'model_selection' in comparison_results:
        ms = comparison_results['model_selection']
        ms_summary = []
        if 'aic' in ms:
            ms_summary.append(f"AIC: {ms['aic']['preferred']} preferred ({ms['aic']['strength']})")
        if 'bic' in ms:
            ms_summary.append(f"BIC: {ms['bic']['preferred']} preferred ({ms['bic']['strength']})")
        
        if ms_summary:
            summary_details.append(f"Model Selection: {' | '.join(ms_summary)}")
    
    # Create the final summary text
    final_summary = ' | '.join(summary_details)
    
    ax8.text(0.5, 0.5, final_summary, ha='center', va='center', 
            transform=ax8.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=1", facecolor=FLARE_PALETTE['accent'], 
                     alpha=0.3, edgecolor=FLARE_PALETTE['dark'], linewidth=2),
            wrap=True)
    ax8.axis('off')
    
    # Overall figure styling
    fig.suptitle(f'Comprehensive Population Comparison: {labels1} vs {labels2}', 
                fontsize=20, fontweight='bold', y=0.98)
    
    # Add professional footer
    footer_text = f"Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | Solar Flare Analysis Module"
    fig.text(0.99, 0.01, footer_text, ha='right', va='bottom', 
            fontsize=8, style='italic', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        logger.info(f"Enhanced comparison plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()


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


def _plot_power_law_analysis(data: np.ndarray, results: PowerLawResults, 
                           fit_object: Any, save_path: Optional[str] = None):
    """Create comprehensive plots for power-law analysis with enhanced styling."""
    # Set up the figure with professional styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20, 14))
    
    # Create custom grid layout
    gs = gridspec.GridSpec(3, 4, height_ratios=[2, 1.5, 1], width_ratios=[2, 1.5, 1.5, 1],
                          hspace=0.35, wspace=0.35)
    
    # Color palette for consistent styling
    colors = sns.color_palette("husl", 8)
    primary_color = FLARE_PALETTE['primary']
    secondary_color = FLARE_PALETTE['secondary']
    
    # 1. Main power-law fit plot (enhanced)
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Plot data histogram with enhanced styling
    data_filtered = data[data >= results.xmin]
    bins = np.logspace(np.log10(data_filtered.min()), np.log10(data_filtered.max()), 50)
    
    # Create histogram with gradient effect
    n, bins_hist, patches = ax1.hist(data_filtered, bins=bins, density=True, alpha=0.7, 
                                    color=primary_color, edgecolor='white', linewidth=0.5,
                                    label='Observed Data')
    
    # Add gradient effect to histogram bars
    for i, p in enumerate(patches):
        alpha_val = 0.3 + 0.7 * (i / len(patches))
        p.set_alpha(alpha_val)
    
    # Plot power-law fit with confidence band
    x_range = np.logspace(np.log10(results.xmin), np.log10(data_filtered.max()), 200)
    y_fit = (results.alpha - 1) / results.xmin * (x_range / results.xmin) ** (-results.alpha)
    
    ax1.plot(x_range, y_fit, color=secondary_color, linewidth=3, 
             label=f'Power-law Fit: α = {results.alpha:.3f} ± {results.alpha_err:.3f}')
    
    # Add confidence band if bootstrap samples available
    if results.bootstrap_samples and len(results.bootstrap_samples) > 10:
        alpha_lower, alpha_upper = np.percentile(results.bootstrap_samples, [2.5, 97.5])
        y_lower = (alpha_lower - 1) / results.xmin * (x_range / results.xmin) ** (-alpha_lower)
        y_upper = (alpha_upper - 1) / results.xmin * (x_range / results.xmin) ** (-alpha_upper)
        ax1.fill_between(x_range, y_lower, y_upper, alpha=0.2, color=secondary_color,
                        label='95% Confidence Band')
    
    # Plot alternative distributions
    dist_colors = list(DISTRIBUTION_PALETTE.values())
    for i, (dist_name, comparison) in enumerate(results.distribution_comparison.items()):
        if not np.isnan(comparison['R']) and i < len(dist_colors):
            try:
                if hasattr(fit_object, dist_name):
                    dist_obj = getattr(fit_object, dist_name)
                    x_dist = np.logspace(np.log10(results.xmin), np.log10(data_filtered.max()), 100)
                    if dist_name == 'lognormal':
                        y_dist = dist_obj.pdf(x_dist)
                    elif dist_name == 'exponential':
                        y_dist = dist_obj.pdf(x_dist)
                    else:
                        continue
                    
                    ax1.plot(x_dist, y_dist, '--', color=dist_colors[i], linewidth=2, alpha=0.8,
                            label=f'{dist_name.replace("_", " ").title()} (R={comparison["R"]:.2f})')
            except Exception as e:
                continue
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Energy', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax1.set_title(f'Power-Law Distribution Analysis\n'
                 f'α = {results.alpha:.3f} ± {results.alpha_err:.3f} | '
                 f'KS = {results.ks_statistic:.4f} | n = {results.n_data:,}',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Enhanced legend
    legend = ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
                       borderpad=1, fontsize=10)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Add grid and styling
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    
    # 2. Bootstrap distribution with KDE
    ax2 = fig.add_subplot(gs[0, 2])
    if results.bootstrap_samples and len(results.bootstrap_samples) > 10:
        # Histogram
        ax2.hist(results.bootstrap_samples, bins=30, density=True, alpha=0.6, 
                color=primary_color, edgecolor='white', linewidth=0.5)
        
        # KDE overlay
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(results.bootstrap_samples)
            x_kde = np.linspace(min(results.bootstrap_samples), max(results.bootstrap_samples), 200)
            ax2.plot(x_kde, kde(x_kde), color=secondary_color, linewidth=3, label='KDE')
        except:
            pass
        
        # Add vertical lines for statistics
        ax2.axvline(results.alpha, color='red', linestyle='-', linewidth=2,
                   label=f'α = {results.alpha:.3f}')
        
        if len(results.bootstrap_samples) > 10:
            ci_lower, ci_upper = np.percentile(results.bootstrap_samples, [2.5, 97.5])
            ax2.axvline(ci_lower, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            ax2.axvline(ci_upper, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            ax2.fill_betweenx([0, ax2.get_ylim()[1]], ci_lower, ci_upper, 
                            alpha=0.2, color='red', label='95% CI')
        
        ax2.set_xlabel('Power-law Exponent (α)', fontweight='bold')
        ax2.set_ylabel('Density', fontweight='bold')
        ax2.set_title('Bootstrap Distribution', fontweight='bold', pad=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Insufficient Bootstrap\nData Available', 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=FLARE_PALETTE['light'], alpha=0.8))
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    # 3. Complementary CDF plot
    ax3 = fig.add_subplot(gs[0, 3])
    data_sorted = np.sort(data_filtered)
    n_data = len(data_sorted)
    empirical_ccdf = np.arange(n_data, 0, -1) / n_data
    
    # Plot empirical CCDF
    ax3.loglog(data_sorted, empirical_ccdf, 'o', color=primary_color, 
              markersize=4, alpha=0.7, markeredgecolor='white', markeredgewidth=0.5,
              label='Empirical CCDF')
    
    # Plot theoretical CCDF
    x_theory = np.logspace(np.log10(results.xmin), np.log10(data_sorted.max()), 100)
    y_theory = (x_theory / results.xmin) ** (1 - results.alpha)
    
    ax3.loglog(x_theory, y_theory, '-', color=secondary_color, linewidth=3,
              label=f'Theoretical (α={results.alpha:.3f})')
    
    ax3.set_xlabel('Energy', fontweight='bold')
    ax3.set_ylabel('P(X ≥ x)', fontweight='bold')
    ax3.set_title('Complementary CDF', fontweight='bold', pad=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Q-Q Plot with enhanced styling
    ax4 = fig.add_subplot(gs[1, 0])
    try:
        n_qq = len(data_sorted)
        theoretical_quantiles = np.linspace(0.01, 0.99, n_qq)
        theoretical_values = results.xmin * (1 - theoretical_quantiles) ** (-1/(results.alpha - 1))
        
        # Scatter plot with color gradient
        scatter = ax4.scatter(theoretical_values, data_sorted, 
                            c=np.arange(len(data_sorted)), cmap='viridis',
                            alpha=0.7, s=20, edgecolors='white', linewidth=0.3)
        
        # Perfect fit line
        min_val, max_val = min(data_sorted.min(), theoretical_values.min()), \
                          max(data_sorted.max(), theoretical_values.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, alpha=0.8,
                label='Perfect Fit')
        
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.set_xlabel('Theoretical Quantiles', fontweight='bold')
        ax4.set_ylabel('Sample Quantiles', fontweight='bold')
        ax4.set_title('Q-Q Plot', fontweight='bold', pad=10)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4, shrink=0.8)
        cbar.set_label('Data Order', fontweight='bold', fontsize=9)
        
    except Exception as e:
        ax4.text(0.5, 0.5, f'Q-Q Plot Failed:\n{str(e)[:30]}...', 
                ha='center', va='center', transform=ax4.transAxes,
                fontweight='bold')
        ax4.set_xticks([])
        ax4.set_yticks([])
    
    # 5. Distribution comparison radar/bar chart
    ax5 = fig.add_subplot(gs[1, 1])
    if results.distribution_comparison:
        dist_names = list(results.distribution_comparison.keys())
        r_values = [results.distribution_comparison[d]['R'] for d in dist_names]
        p_values = [results.distribution_comparison[d]['p'] for d in dist_names]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(dist_names))
        colors_bar = [DISTRIBUTION_PALETTE.get(name, colors[i % len(colors)]) 
                     for i, name in enumerate(dist_names)]
        
        bars = ax5.barh(y_pos, r_values, color=colors_bar, alpha=0.8, 
                       edgecolor='white', linewidth=1)
        
        # Add value labels
        for i, (bar, r_val, p_val) in enumerate(zip(bars, r_values, p_values)):
            width = bar.get_width()
            label_x = width + 0.1 if width >= 0 else width - 0.1
            ha = 'left' if width >= 0 else 'right'
            ax5.text(label_x, bar.get_y() + bar.get_height()/2,
                    f'R={r_val:.2f}\np={p_val:.3f}' if not np.isnan(p_val) else f'R={r_val:.2f}',
                    ha=ha, va='center', fontsize=8, fontweight='bold')
        
        ax5.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels([name.replace('_', ' ').title() for name in dist_names])
        ax5.set_xlabel('Log-likelihood Ratio (R)', fontweight='bold')
        ax5.set_title('Distribution Comparison\n(R > 0: Power-law Preferred)', 
                     fontweight='bold', pad=10)
        ax5.grid(True, alpha=0.3, axis='x')
    else:
        ax5.text(0.5, 0.5, 'No Distribution\nComparisons Available', 
                ha='center', va='center', transform=ax5.transAxes,
                fontsize=12, fontweight='bold')
        ax5.set_xticks([])
        ax5.set_yticks([])
    
    # 6. Residuals plot
    ax6 = fig.add_subplot(gs[1, 2])
    try:
        # Calculate residuals
        x_data = data_sorted[data_sorted >= results.xmin]
        empirical_cdf = np.arange(1, len(x_data) + 1) / len(x_data)
        theoretical_cdf = 1 - (x_data / results.xmin) ** (1 - results.alpha)
        residuals = empirical_cdf - theoretical_cdf
        
        # Residuals plot
        ax6.scatter(x_data, residuals, alpha=0.6, s=15, color=primary_color,
                   edgecolors='white', linewidth=0.3)
        ax6.axhline(y=0, color='red', linestyle='-', linewidth=2, alpha=0.8)
        
        # Add running average
        if len(residuals) > 10:
            window_size = max(5, len(residuals) // 20)
            running_avg = pd.Series(residuals).rolling(window=window_size, center=True).mean()
            ax6.plot(x_data, running_avg, color=secondary_color, linewidth=2,
                    label=f'Running Average (n={window_size})')
            ax6.legend(fontsize=9)
        
        ax6.set_xscale('log')
        ax6.set_xlabel('Energy', fontweight='bold')
        ax6.set_ylabel('Residuals (Empirical - Theoretical)', fontweight='bold')
        ax6.set_title('Model Residuals', fontweight='bold', pad=10)
        ax6.grid(True, alpha=0.3)
        
    except Exception as e:
        ax6.text(0.5, 0.5, f'Residuals Plot Failed:\n{str(e)[:30]}...', 
                ha='center', va='center', transform=ax6.transAxes,
                fontweight='bold')
        ax6.set_xticks([])
        ax6.set_yticks([])
    
    # 7. Goodness-of-fit summary panel
    ax7 = fig.add_subplot(gs[1, 3])
    
    # Create summary statistics
    summary_stats = []
    summary_stats.append(f"Power-law Exponent")
    summary_stats.append(f"α = {results.alpha:.3f} ± {results.alpha_err:.3f}")
    summary_stats.append("")
    summary_stats.append(f"Data Range")
    summary_stats.append(f"xmin = {results.xmin:.2e}")
    summary_stats.append(f"xmax = {data.max():.2e}")
    summary_stats.append("")
    summary_stats.append(f"Sample Statistics")
    summary_stats.append(f"n = {results.n_data:,} points")
    summary_stats.append(f"KS = {results.ks_statistic:.4f}")
    
    if not np.isnan(results.p_value):
        summary_stats.append(f"p-value = {results.p_value:.4f}")
    
    if not np.isnan(results.aic):
        summary_stats.append("")
        summary_stats.append(f"Information Criteria")
        summary_stats.append(f"AIC = {results.aic:.2f}")
        summary_stats.append(f"BIC = {results.bic:.2f}")
    
    # Create text box with enhanced styling
    text_content = '\n'.join(summary_stats)
    ax7.text(0.05, 0.95, text_content, transform=ax7.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.8", facecolor=FLARE_PALETTE['background'], 
                     edgecolor=FLARE_PALETTE['dark'], linewidth=1.5, alpha=0.9))
    ax7.axis('off')
    
    # 8. Energy distribution histogram (bottom panel)
    ax8 = fig.add_subplot(gs[2, :])
    
       
    # Create multi-scale histogram
    fig_hist, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Linear scale histogram
    ax_lin.hist(data, bins=50, alpha=0.7, color=primary_color, 
               edgecolor='white', linewidth=0.5, density=True)
    ax_lin.set_xlabel('Energy (Linear Scale)', fontweight='bold')
    ax_lin.set_ylabel('Density', fontweight='bold')
    ax_lin.set_title('Energy Distribution (Linear)', fontweight='bold')
    ax_lin.grid(True, alpha=0.3)
    
    # Log scale histogram
    log_bins = np.logspace(np.log10(data[data > 0].min()), np.log10(data.max()), 50)
    ax_log.hist(data[data > 0], bins=log_bins, alpha=0.7, color=secondary_color,
               edgecolor='white', linewidth=0.5, density=True)
    ax_log.set_xscale('log')
    ax_log.set_yscale('log')
    ax_log.set_xlabel('Energy (Log Scale)', fontweight='bold')
    ax_log.set_ylabel('Density', fontweight='bold')
    ax_log.set_title('Energy Distribution (Log)', fontweight='bold')
    ax_log.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.close(fig_hist)  # Close the temporary figure
    
    # Add summary text to bottom panel
    summary_text = (f"Power-Law Analysis Summary | "
                   f"α = {results.alpha:.3f} ± {results.alpha_err:.3f} | "
                   f"Data: {results.n_data:,} points | "
                   f"Range: [{results.xmin:.2e}, {data.max():.2e}] | "
                   f"KS Statistic: {results.ks_statistic:.4f}")
    
    ax8.text(0.5, 0.5, summary_text, ha='center', va='center', 
            transform=ax8.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=1", facecolor=FLARE_PALETTE['accent'], 
                     alpha=0.2, edgecolor=FLARE_PALETTE['dark'], linewidth=1))
    ax8.axis('off')
    
    # Overall figure styling
    fig.suptitle('Comprehensive Power-Law Analysis Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Add professional footer
    footer_text = f"Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | Solar Flare Analysis Module"
    fig.text(0.99, 0.01, footer_text, ha='right', va='bottom', 
            fontsize=8, style='italic', alpha=0.7)
    
    # Save with high quality
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        logger.info(f"Enhanced plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()


def create_interactive_power_law_plot(data: np.ndarray, results: PowerLawResults,
                                    title: str = "Interactive Power-Law Analysis") -> None:
    """
    Create interactive power-law analysis plots using Plotly.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    results : PowerLawResults
        Fitting results
    title : str
        Plot title
    """
    if not HAS_PLOTLY:
        logger.warning("Plotly not available. Falling back to static plots.")
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=['Power-Law Fit', 'Bootstrap Distribution', 'Q-Q Plot',
                       'Complementary CDF', 'Distribution Comparison', 'Residuals'],
        specs=[[{"type": "log"}, {"type": "xy"}, {"type": "log"}],
               [{"type": "log"}, {"type": "xy"}, {"type": "xy"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # 1. Main power-law fit
    data_filtered = data[data >= results.xmin]
    bins = np.logspace(np.log10(data_filtered.min()), np.log10(data_filtered.max()), 50)
    hist, bin_edges = np.histogram(data_filtered, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    # Data histogram
    fig.add_trace(
        go.Scatter(x=bin_centers, y=hist, mode='markers', name='Data',
                  marker=dict(color=FLARE_PALETTE['primary'], size=6, opacity=0.7),
                  hovertemplate='Energy: %{x:.2e}<br>Density: %{y:.2e}<extra></extra>'),
        row=1, col=1
    )
    
    # Power-law fit
    x_fit = np.logspace(np.log10(results.xmin), np.log10(data_filtered.max()), 200)
    y_fit = (results.alpha - 1) / results.xmin * (x_fit / results.xmin) ** (-results.alpha)
    
    fig.add_trace(
        go.Scatter(x=x_fit, y=y_fit, mode='lines', name=f'Power-law (α={results.alpha:.3f})',
                  line=dict(color=FLARE_PALETTE['secondary'], width=3),
                  hovertemplate='Energy: %{x:.2e}<br>PDF: %{y:.2e}<extra></extra>'),
        row=1, col=1
    )
    
    # 2. Bootstrap distribution
    if results.bootstrap_samples:
        fig.add_trace(
            go.Histogram(x=results.bootstrap_samples, nbinsx=30, name='Bootstrap',
                        marker_color=FLARE_PALETTE['tertiary'], opacity=0.7,
                        hovertemplate='α: %{x:.3f}<br>Count: %{y}<extra></extra>'),
            row=1, col=2
        )
        
        # Add mean line
        fig.add_vline(x=results.alpha, line_dash="dash", line_color="red",
                     annotation_text=f"α = {results.alpha:.3f}", row=1, col=2)
    
    # 3. Q-Q Plot
    try:
        data_sorted = np.sort(data_filtered)
        n = len(data_sorted)
        theoretical_quantiles = np.linspace(0.01, 0.99, n)
        theoretical_values = results.xmin * (1 - theoretical_quantiles) ** (-1/(results.alpha - 1))
        
        fig.add_trace(
            go.Scatter(x=theoretical_values, y=data_sorted, mode='markers',
                      name='Q-Q Points', marker=dict(size=4, opacity=0.6),
                      hovertemplate='Theoretical: %{x:.2e}<br>Observed: %{y:.2e}<extra></extra>'),
            row=1, col=3
        )
        
        # Perfect fit line
        min_val = min(data_sorted.min(), theoretical_values.min())
        max_val = max(data_sorted.max(), theoretical_values.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                      name='Perfect Fit', line=dict(color='red', dash='dash')),
            row=1, col=3
        )
    except:
        pass
    
    # 4. Complementary CDF
    data_sorted = np.sort(data_filtered)
    ccdf = np.arange(len(data_sorted), 0, -1) / len(data_sorted)
    
    fig.add_trace(
        go.Scatter(x=data_sorted, y=ccdf, mode='markers', name='Empirical CCDF',
                  marker=dict(size=4, opacity=0.7),
                  hovertemplate='Energy: %{x:.2e}<br>P(X≥x): %{y:.2e}<extra></extra>'),
        row=2, col=1
    )
    
    # Theoretical CCDF
    x_theory = np.logspace(np.log10(results.xmin), np.log10(data_sorted.max()), 100)
    y_theory = (x_theory / results.xmin) ** (1 - results.alpha)
    
    fig.add_trace(
        go.Scatter(x=x_theory, y=y_theory, mode='lines', name='Theoretical CCDF',
                  line=dict(color=FLARE_PALETTE['secondary'], width=3)),
        row=2, col=1
    )
    
    # 5. Distribution comparison
    if results.distribution_comparison:
        dist_names = list(results.distribution_comparison.keys())
        r_values = [results.distribution_comparison[d]['R'] for d in dist_names]
        
        fig.add_trace(
            go.Bar(x=dist_names, y=r_values, name='Likelihood Ratio',
                  marker_color=FLARE_PALETTE['quaternary'],
                  hovertemplate='Distribution: %{x}<br>R: %{y:.3f}<extra></extra>'),
            row=2, col=2
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
    
    # 6. Residuals
    try:
        empirical_cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
        theoretical_cdf = 1 - (data_sorted / results.xmin) ** (1 - results.alpha)
        residuals = empirical_cdf - theoretical_cdf
        
        fig.add_trace(
            go.Scatter(x=data_sorted, y=residuals, mode='markers', name='Residuals',
                      marker=dict(size=4, opacity=0.6),
                      hovertemplate='Energy: %{x:.2e}<br>Residual: %{y:.3f}<extra></extra>'),
            row=2, col=3
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=3)
    except:
        pass
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16, family="Arial Black")),
        height=800,
        showlegend=True,
        template="plotly_white",
        font=dict(family="Arial", size=12)
    )
    
    # Update axes
    fig.update_xaxes(title_text="Energy", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Probability Density", type="log", row=1, col=1)
    
    fig.update_xaxes(title_text="Power-law Exponent (α)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    fig.update_xaxes(title_text="Theoretical Quantiles", type="log", row=1, col=3)
    fig.update_yaxes(title_text="Sample Quantiles", type="log", row=1, col=3)
    
    fig.update_xaxes(title_text="Energy", type="log", row=2, col=1)
    fig.update_yaxes(title_text="P(X ≥ x)", type="log", row=2, col=1)
    
    fig.update_xaxes(title_text="Distribution", row=2, col=2)
    fig.update_yaxes(title_text="Log-likelihood Ratio", row=2, col=2)
    
    fig.update_xaxes(title_text="Energy", type="log", row=2, col=3)
    fig.update_yaxes(title_text="Residuals", row=2, col=3)
    
    fig.show()
    
    return fig


def create_publication_ready_plot(data: np.ndarray, results: PowerLawResults,
                                 save_path: Optional[str] = None,
                                 figure_size: Tuple[float, float] = (12, 8),
                                 dpi: int = 300) -> None:
    """
    Create publication-ready power-law plots with minimal, clean styling.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    results : PowerLawResults
        Fitting results
    save_path : str, optional
        Path to save the figure
    figure_size : tuple
        Figure size in inches
    dpi : int
        Resolution for saved figure
    """
    # Set publication style
    with plt.style.context(['seaborn-v0_8-paper']):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figure_size)
        
        # Filter data
        data_filtered = data[data >= results.xmin]
        
        # 1. Main fit plot
        bins = np.logspace(np.log10(data_filtered.min()), np.log10(data_filtered.max()), 40)
        ax1.hist(data_filtered, bins=bins, density=True, alpha=0.7, 
                color='lightblue', edgecolor='navy', linewidth=0.5, label='Data')
        
        # Power-law fit
        x_fit = np.logspace(np.log10(results.xmin), np.log10(data_filtered.max()), 200)
        y_fit = (results.alpha - 1) / results.xmin * (x_fit / results.xmin) ** (-results.alpha)
        ax1.plot(x_fit, y_fit, 'r-', linewidth=2, 
                label=f'Power law: α = {results.alpha:.2f} ± {results.alpha_err:.2f}')
        
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Energy', fontweight='bold')
        ax1.set_ylabel('Probability density', fontweight='bold')
        ax1.legend(frameon=True, fancybox=True)
        ax1.grid(True, alpha=0.3)
        ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes, fontsize=14, 
                fontweight='bold', va='top')
        
        # 2. Complementary CDF
        data_sorted = np.sort(data_filtered)
        ccdf = np.arange(len(data_sorted), 0, -1) / len(data_sorted)
        
        ax2.loglog(data_sorted, ccdf, 'o', markersize=3, alpha=0.6, 
                  color='blue', markeredgewidth=0, label='Empirical')
        
        # Theoretical CCDF
        x_theory = np.logspace(np.log10(results.xmin), np.log10(data_sorted.max()), 100)
        y_theory = (x_theory / results.xmin) ** (1 - results.alpha)
        ax2.loglog(x_theory, y_theory, 'r-', linewidth=2, label='Theoretical')
        
        ax2.set_xlabel('Energy', fontweight='bold')
        ax2.set_ylabel('P(X ≥ x)', fontweight='bold')
        ax2.legend(frameon=True, fancybox=True)
        ax2.grid(True, alpha=0.3)
        ax2.text(0.05, 0.95, '(b)', transform=ax2.transAxes, fontsize=14, 
                fontweight='bold', va='top')
        
        # 3. Bootstrap distribution (if available)
        if results.bootstrap_samples and len(results.bootstrap_samples) > 10:
            ax3.hist(results.bootstrap_samples, bins=25, density=True, alpha=0.7,
                    color='lightcoral', edgecolor='darkred', linewidth=0.5)
            ax3.axvline(results.alpha, color='red', linestyle='--', linewidth=2,
                       label=f'α = {results.alpha:.3f}')
            
            # Add confidence interval
            ci_lower, ci_upper = np.percentile(results.bootstrap_samples, [2.5, 97.5])
            ax3.axvline(ci_lower, color='red', linestyle=':', alpha=0.7)
            ax3.axvline(ci_upper, color='red', linestyle=':', alpha=0.7)
            ax3.fill_betweenx([0, ax3.get_ylim()[1]], ci_lower, ci_upper, 
                            alpha=0.2, color='red', label='95% CI')
            
            ax3.set_xlabel('Power-law exponent (α)', fontweight='bold')
            ax3.set_ylabel('Density', fontweight='bold')
            ax3.legend(frameon=True, fancybox=True)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Bootstrap analysis\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=12, style='italic')
            ax3.set_xticks([])
            ax3.set_yticks([])
        
        ax3.text(0.05, 0.95, '(c)', transform=ax3.transAxes, fontsize=14, 
                fontweight='bold', va='top')
        
        # 4. Summary statistics
        ax4.axis('off')
        
        # Create summary table
        summary_data = [
            ['Parameter', 'Value'],
            ['Power-law exponent (α)', f'{results.alpha:.3f} ± {results.alpha_err:.3f}'],
            ['Lower bound (x_min)', f'{results.xmin:.2e}'],
            ['Sample size', f'{results.n_data:,}'],
            ['KS statistic', f'{results.ks_statistic:.4f}'],
        ]
        
        if not np.isnan(results.p_value):
            summary_data.append(['p-value', f'{results.p_value:.4f}'])
        
        if not np.isnan(results.aic):
            summary_data.append(['AIC', f'{results.aic:.1f}'])
            summary_data.append(['BIC', f'{results.bic:.1f}'])
        
        # Create table
        table = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0],
                         cellLoc='left', loc='center', colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(summary_data)):
            if i == 0:  # Header
                for j in range(2):
                    table[(i, j)].set_facecolor('#4CAF50')
                    table[(i, j)].set_text_props(weight='bold', color='white')
            else:  # Data rows
                for j in range(2):
                    table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax4.text(0.05, 0.95, '(d)', transform=ax4.transAxes, fontsize=14, 
                fontweight='bold', va='top')
        
        # Overall styling
        plt.tight_layout(pad=3.0)
        
        # Add main title
        fig.suptitle('Power-Law Distribution Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            logger.info(f"Publication-ready plot saved to {save_path}")
        
        plt.show()


def create_dashboard_visualization(data_dict: Dict[str, np.ndarray],
                                 results_dict: Dict[str, PowerLawResults],
                                 save_path: Optional[str] = None) -> None:
    """
    Create a comprehensive dashboard for multiple datasets.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary with dataset names as keys and data arrays as values
    results_dict : dict
        Dictionary with dataset names as keys and PowerLawResults as values
    save_path : str, optional
        Path to save the dashboard
    """
    n_datasets = len(data_dict)
    if n_datasets == 0:
        logger.error("No datasets provided for dashboard")
        return
    
    # Set up the dashboard layout
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 4, height_ratios=[2, 1.5, 1], hspace=0.35, wspace=0.3)
    
    # Color palette for different datasets
    colors = sns.color_palette("husl", n_datasets)
    
    # 1. Combined comparison plot
    ax1 = fig.add_subplot(gs[0, :2])
    
    for i, (name, data) in enumerate(data_dict.items()):
        if name in results_dict:
            results = results_dict[name]
            data_filtered = data[data >= results.xmin]
            
            # Plot histogram
            bins = np.logspace(np.log10(data_filtered.min()), np.log10(data_filtered.max()), 40)
            ax1.hist(data_filtered, bins=bins, density=True, alpha=0.6,
                    color=colors[i], label=f'{name} (α={results.alpha:.3f})',
                    edgecolor='white', linewidth=0.5)
            
            # Plot fit
            x_fit = np.logspace(np.log10(results.xmin), np.log10(data_filtered.max()), 100)
            y_fit = (results.alpha - 1) / results.xmin * (x_fit / results.xmin) ** (-results.alpha)
            ax1.plot(x_fit, y_fit, '--', color=colors[i], linewidth=2, alpha=0.8)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Energy', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Probability Density', fontweight='bold', fontsize=12)
    ax1.set_title('Multi-Dataset Power-Law Comparison', fontweight='bold', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Alpha comparison
    ax2 = fig.add_subplot(gs[0, 2])
    
    names = list(results_dict.keys())
    alphas = [results_dict[name].alpha for name in names]
    alpha_errs = [results_dict[name].alpha_err for name in names]
    
    bars = ax2.bar(range(len(names)), alphas, yerr=alpha_errs, capsize=5,
                   color=colors[:len(names)], alpha=0.8, edgecolor='white', linewidth=1)
    
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('Power-law Exponent (α)', fontweight='bold')
    ax2.set_title('Exponent Comparison', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, alpha, err) in enumerate(zip(bars, alphas, alpha_errs)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.01,
                f'{alpha:.3f}±{err:.3f}', ha='center', va='bottom', fontsize=9, rotation=45)
    
    # 3. Sample size comparison
    ax3 = fig.add_subplot(gs[0, 3])
    
    sample_sizes = [results_dict[name].n_data for name in names]
    bars = ax3.bar(range(len(names)), sample_sizes, color=colors[:len(names)], 
                   alpha=0.8, edgecolor='white', linewidth=1)
    
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=45, ha='right')
    ax3.set_ylabel('Sample Size', fontweight='bold')
    ax3.set_title('Dataset Sizes', fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, size in zip(bars, sample_sizes):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{size:,}', ha='center', va='bottom', fontsize=9, rotation=45)
    
    # 4. Goodness-of-fit comparison
    ax4 = fig.add_subplot(gs[1, :2])
    
    ks_stats = [results_dict[name].ks_statistic for name in names]
    p_values = [results_dict[name].p_value if not np.isnan(results_dict[name].p_value) else 0 
               for name in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, ks_stats, width, label='KS Statistic', 
                    color=FLARE_PALETTE['primary'], alpha=0.8)
    bars2 = ax4.bar(x + width/2, p_values, width, label='p-value', 
                    color=FLARE_PALETTE['secondary'], alpha=0.8)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=45, ha='right')
    ax4.set_ylabel('Value', fontweight='bold')
    ax4.set_title('Goodness-of-Fit Statistics', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Energy range comparison
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Box plot of energy ranges
    energy_ranges = []
    labels_ranges = []
    
    for name, data in data_dict.items():
        if name in results_dict:
            results = results_dict[name]
            data_filtered = data[data >= results.xmin]
            energy_ranges.append(data_filtered)
            labels_ranges.append(name)
    
    if energy_ranges:
        bp = ax5.boxplot(energy_ranges, labels=labels_ranges, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors[:len(energy_ranges)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('white')
            patch.set_linewidth(1)
    
    ax5.set_yscale('log')
    ax5.set_ylabel('Energy Range', fontweight='bold')
    ax5.set_title('Energy Distribution\nComparison', fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Summary statistics table
    ax6 = fig.add_subplot(gs[1, 3])
    ax6.axis('off')
    
    # Create comprehensive summary
    table_data = [['Dataset', 'α', 'x_min', 'n', 'KS']]
    
    for name in names:
        results = results_dict[name]
        row = [
            name[:10] + '...' if len(name) > 10 else name,
            f'{results.alpha:.3f}',
            f'{results.xmin:.1e}',
            f'{results.n_data:,}',
            f'{results.ks_statistic:.3f}'
        ]
        table_data.append(row)
    
    table = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center', colWidths=[0.25, 0.15, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Style the table
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            if i == 0:  # Header
                table[(i, j)].set_facecolor(FLARE_PALETTE['primary'])
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:  # Data rows
                table[(i, j)].set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
    
    # 7. Bottom summary panel
    ax7 = fig.add_subplot(gs[2, :])
    
    # Create overall summary
    total_samples = sum(results_dict[name].n_data for name in names)
    avg_alpha = np.mean([results_dict[name].alpha for name in names])
    alpha_std = np.std([results_dict[name].alpha for name in names])
    
    summary_text = (f"MULTI-DATASET POWER-LAW ANALYSIS DASHBOARD | "
                   f"Datasets: {len(names)} | "
                   f"Total Samples: {total_samples:,} | "
                   f"Average α: {avg_alpha:.3f} ± {alpha_std:.3f} | "
                   f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    ax7.text(0.5, 0.5, summary_text, ha='center', va='center', 
            transform=ax7.transAxes, fontsize=14, fontweight='bold',
            bbox=dict(boxstyle="round,pad=1", facecolor=FLARE_PALETTE['accent'], 
                     alpha=0.3, edgecolor=FLARE_PALETTE['dark'], linewidth=2))
    ax7.axis('off')
    
    # Overall styling
    fig.suptitle('Power-Law Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        logger.info(f"Dashboard saved to {save_path}")
    
    plt.tight_layout()
    plt.show()

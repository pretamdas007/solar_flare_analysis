"""
Enhanced analysis module for flare energy statistics and power-law distributions.

This module provides comprehensive tools for analyzing solar flare energy distributions,
including robust power-law fitting, statistical testing, and comparative analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize, integrate
from scipy.interpolate import interp1d
import powerlaw
import warnings
from typing import Optional, Union, Tuple, Dict, List, Any
from dataclasses import dataclass
from pathlib import Path
import logging

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

# Set style for better plots
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')


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
    """Create comprehensive comparison plots."""
    fig = plt.figure(figsize=(16, 12))
    
    # Main comparison plot
    ax1 = plt.subplot(2, 3, (1, 2))
    
    # Plot histograms and fits
    bins = np.logspace(np.log10(min(np.min(energies1), np.min(energies2))),
                      np.log10(max(np.max(energies1), np.max(energies2))), 50)
    
    plt.hist(energies1, bins=bins, alpha=0.6, density=True, label=f'{labels1} Data', color='blue')
    plt.hist(energies2, bins=bins, alpha=0.6, density=True, label=f'{labels2} Data', color='red')
    
    # Plot power-law fits
    x_range = np.logspace(np.log10(max(fit1.xmin, fit2.xmin)), 
                         np.log10(min(np.max(energies1), np.max(energies2))), 100)
    
    # Power-law PDFs
    y1 = (fit1.alpha - 1) / fit1.xmin * (x_range / fit1.xmin) ** (-fit1.alpha)
    y2 = (fit2.alpha - 1) / fit2.xmin * (x_range / fit2.xmin) ** (-fit2.alpha)
    
    plt.loglog(x_range, y1, '--', color='blue', linewidth=2,
              label=f'{labels1} Fit: α = {fit1.alpha:.3f} ± {fit1.alpha_err:.3f}')
    plt.loglog(x_range, y2, '--', color='red', linewidth=2,
              label=f'{labels2} Fit: α = {fit2.alpha:.3f} ± {fit2.alpha_err:.3f}')
    
    plt.xlabel('Energy')
    plt.ylabel('Probability Density')
    plt.title(f'Population Comparison\nΔα = {comparison_results["alpha_difference"]:.3f} '
             f'({comparison_results["significance_sigma"]:.2f}σ)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Bootstrap comparison
    ax2 = plt.subplot(2, 3, 3)
    if fit1.bootstrap_samples and fit2.bootstrap_samples:
        plt.hist(fit1.bootstrap_samples, bins=30, alpha=0.6, color='blue', 
                label=f'{labels1}', density=True)
        plt.hist(fit2.bootstrap_samples, bins=30, alpha=0.6, color='red', 
                label=f'{labels2}', density=True)
        plt.axvline(fit1.alpha, color='blue', linestyle='--', linewidth=2)
        plt.axvline(fit2.alpha, color='red', linestyle='--', linewidth=2)
        plt.xlabel('α')
        plt.ylabel('Density')
        plt.title('Bootstrap Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Bootstrap data\nnot available', 
                ha='center', va='center', transform=ax2.transAxes)
    
    # Statistical tests summary
    ax3 = plt.subplot(2, 3, 4)
    test_text = []
    test_text.append(f"Population Comparison: {labels1} vs {labels2}")
    test_text.append(f"Sample sizes: {fit1.n_data} vs {fit2.n_data}")
    test_text.append(f"Alpha difference: {comparison_results['alpha_difference']:.4f}")
    test_text.append(f"Significance: {comparison_results['significance_sigma']:.2f}σ")
    
    if 'statistical_tests' in comparison_results:
        tests = comparison_results['statistical_tests']
        if 'ks_test' in tests:
            ks = tests['ks_test']
            test_text.append(f"KS test: D={ks['statistic']:.4f}, p={ks['p_value']:.4f}")
        
        if 'anderson_test' in tests:
            ad = tests['anderson_test']
            test_text.append(f"Anderson-Darling: {ad['statistic']:.4f}")
    
    plt.text(0.05, 0.95, '\n'.join(test_text), transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    plt.axis('off')
    
    # Cumulative distribution comparison
    ax4 = plt.subplot(2, 3, 5)
    x1_sorted = np.sort(energies1)
    x2_sorted = np.sort(energies2)
    y1 = np.arange(1, len(x1_sorted) + 1) / len(x1_sorted)
    y2 = np.arange(1, len(x2_sorted) + 1) / len(x2_sorted)
    
    plt.loglog(x1_sorted, 1 - y1, color='blue', label=f'{labels1}', linewidth=2)
    plt.loglog(x2_sorted, 1 - y2, color='red', label=f'{labels2}', linewidth=2)
    
    # Add theoretical power-law lines
    x_theory = np.logspace(np.log10(max(fit1.xmin, fit2.xmin)), 
                          np.log10(min(np.max(energies1), np.max(energies2))), 100)
    y1_theory = (x_theory / fit1.xmin) ** (1 - fit1.alpha)
    y2_theory = (x_theory / fit2.xmin) ** (1 - fit2.alpha)
    
    plt.loglog(x_theory, y1_theory, '--', color='blue', alpha=0.7)
    plt.loglog(x_theory, y2_theory, '--', color='red', alpha=0.7)
    
    plt.xlabel('Energy')
    plt.ylabel('P(X ≥ x)')
    plt.title('Complementary CDF Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Model selection summary
    ax5 = plt.subplot(2, 3, 6)
    if 'model_selection' in comparison_results:
        ms = comparison_results['model_selection']
        ms_text = ["Model Selection Results:"]
        
        if 'aic' in ms:
            aic = ms['aic']
            ms_text.append(f"AIC: {aic['preferred']} preferred")
            ms_text.append(f"Evidence: {aic['strength']}")
            ms_text.append(f"ΔAIC: {aic['difference']:.2f}")
        
        if 'bic' in ms:
            bic = ms['bic']
            ms_text.append(f"BIC: {bic['preferred']} preferred")
            ms_text.append(f"Evidence: {bic['strength']}")
            ms_text.append(f"ΔBIC: {bic['difference']:.2f}")
        
        # Add energy statistics comparison
        if 'energy_comparison' in comparison_results:
            ec = comparison_results['energy_comparison']
            if 'ratios' in ec:
                ratios = ec['ratios']
                ms_text.append(f"\nEnergy Ratios ({labels1}/{labels2}):")
                ms_text.append(f"Mean: {ratios['mean_ratio']:.2f}")
                ms_text.append(f"Median: {ratios['median_ratio']:.2f}")
                ms_text.append(f"Range overlap: {ratios['range_overlap']:.2f}")
        
        plt.text(0.05, 0.95, '\n'.join(ms_text), transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")
    
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
    """Create comprehensive plots for power-law analysis."""
    fig = plt.figure(figsize=(15, 12))
    
    # Main fit plot
    ax1 = plt.subplot(2, 3, (1, 2))
    fit_object.plot_pdf(color='blue', alpha=0.7, label='Data')
    fit_object.power_law.plot_pdf(color='red', linestyle='--', linewidth=2, 
                                 label=f'Power-law Fit (α = {results.alpha:.3f})')
    
    # Add alternative distributions if available
    colors = ['green', 'orange', 'purple']
    for i, (dist_name, comparison) in enumerate(results.distribution_comparison.items()):
        if not np.isnan(comparison['R']) and i < len(colors):
            try:
                getattr(fit_object, dist_name).plot_pdf(color=colors[i], linestyle=':', 
                                                       label=f'{dist_name.title()} (R={comparison["R"]:.3f})')
            except:
                pass
    
    plt.title(f'Power-law Analysis\nα = {results.alpha:.3f} ± {results.alpha_err:.3f}\n'
             f'KS = {results.ks_statistic:.4f}, n = {results.n_data}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    
    # Bootstrap distribution
    ax2 = plt.subplot(2, 3, 3)
    if results.bootstrap_samples:
        plt.hist(results.bootstrap_samples, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(results.alpha, color='red', linestyle='--', linewidth=2,
                   label=f'α = {results.alpha:.3f}')
        
        # Add confidence intervals
        if len(results.bootstrap_samples) > 10:
            ci_lower, ci_upper = np.percentile(results.bootstrap_samples, [2.5, 97.5])
            plt.axvline(ci_lower, color='red', linestyle=':', alpha=0.7, label='95% CI')
            plt.axvline(ci_upper, color='red', linestyle=':', alpha=0.7)
            plt.fill_betweenx([0, plt.ylim()[1]], ci_lower, ci_upper, 
                            alpha=0.2, color='red')
        
        plt.title('Bootstrap Distribution')
        plt.xlabel('α')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Insufficient data\nfor bootstrap analysis', 
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    # Q-Q plot
    ax3 = plt.subplot(2, 3, 4)
    try:
        data_sorted = np.sort(data)
        n = len(data_sorted)
        theoretical_quantiles = np.linspace(0.01, 0.99, n)
        
        # Calculate theoretical quantiles for power-law
        theoretical_values = results.xmin * (1 - theoretical_quantiles) ** (-1/(results.alpha - 1))
        
        plt.scatter(theoretical_values, data_sorted, alpha=0.6, s=20)
        plt.plot([data_sorted.min(), data_sorted.max()], 
                [data_sorted.min(), data_sorted.max()], 'r--', linewidth=2)
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Sample Quantiles')
        plt.title('Q-Q Plot')
        plt.grid(True, alpha=0.3)
        plt.loglog()
    except Exception as e:
        plt.text(0.5, 0.5, f'Q-Q plot failed:\n{str(e)[:50]}...', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # Distribution comparison
    ax4 = plt.subplot(2, 3, 5)
    if results.distribution_comparison:
        dist_names = list(results.distribution_comparison.keys())
        r_values = [results.distribution_comparison[d]['R'] for d in dist_names]
        p_values = [results.distribution_comparison[d]['p'] for d in dist_names]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(dist_names)))
        bars = plt.bar(range(len(dist_names)), r_values, color=colors, alpha=0.7)
        
        # Add p-value annotations
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'p={p_val:.3f}' if not np.isnan(p_val) else 'p=N/A',
                    ha='center', va='bottom', fontsize=8)
        
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Alternative Distributions')
        plt.ylabel('Log-likelihood Ratio (R)')
        plt.title('Distribution Comparison\n(R > 0: power-law favored)')
        plt.xticks(range(len(dist_names)), [d.replace('_', '\n') for d in dist_names], 
                  rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No distribution\ncomparisons available', 
                ha='center', va='center', transform=ax4.transAxes)
    
    # Goodness-of-fit summary
    ax5 = plt.subplot(2, 3, 6)
    gof_text = []
    gof_text.append(f"Power-law Exponent: {results.alpha:.3f} ± {results.alpha_err:.3f}")
    gof_text.append(f"Data Range: [{results.xmin:.2e}, {data.max():.2e}]")
    gof_text.append(f"Sample Size: {results.n_data}")
    gof_text.append(f"KS Statistic: {results.ks_statistic:.4f}")
    
    if not np.isnan(results.p_value):
        gof_text.append(f"p-value: {results.p_value:.4f}")
    
    if not np.isnan(results.aic):
        gof_text.append(f"AIC: {results.aic:.2f}")
        gof_text.append(f"BIC: {results.bic:.2f}")
    
    # Add distribution comparison summary
    if results.distribution_comparison:
        gof_text.append("\nAlternative Distributions:")
        for dist, comp in results.distribution_comparison.items():
            if not np.isnan(comp['R']):
                preference = "Power-law" if comp['R'] > 0 else dist.title()
                gof_text.append(f"vs {dist.title()}: {preference}")
    
    plt.text(0.05, 0.95, '\n'.join(gof_text), transform=ax5.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def calculate_flare_statistics(flare_data: pd.DataFrame, 
                             energy_column: str = 'energy',
                             duration_column: Optional[str] = None,
                             peak_flux_column: Optional[str] = None,
                             time_column: Optional[str] = None) -> FlareStatistics:
    """
    Calculate comprehensive statistics for a flare dataset.
    
    Parameters
    ----------
    flare_data : pd.DataFrame
        DataFrame containing flare data
    energy_column : str, default 'energy'
        Name of energy column
    duration_column : str, optional
        Name of duration column
    peak_flux_column : str, optional
        Name of peak flux column
    time_column : str, optional
        Name of time column for temporal analysis
        
    Returns
    -------
    FlareStatistics
        Comprehensive statistics object
    """
    energies = flare_data[energy_column].dropna()
    energies = energies[energies > 0]  # Remove non-positive energies
    
    if len(energies) == 0:
        logger.warning("No valid energy data found")
        return FlareStatistics(
            n_flares=0, total_energy=0, mean_energy=0, median_energy=0,
            energy_std=0, energy_range=(0, 0), duration_stats={},
            peak_flux_stats={}, temporal_distribution={}
        )
    
    # Basic energy statistics
    energy_stats = {
        'n_flares': len(energies),
        'total_energy': energies.sum(),
        'mean_energy': energies.mean(),
        'median_energy': energies.median(),
        'energy_std': energies.std(),
        'energy_range': (energies.min(), energies.max())
    }
    
    # Duration statistics
    duration_stats = {}
    if duration_column and duration_column in flare_data.columns:
        durations = flare_data[duration_column].dropna()
        if len(durations) > 0:
            duration_stats = {
                'mean_duration': durations.mean(),
                'median_duration': durations.median(),
                'std_duration': durations.std(),
                'min_duration': durations.min(),
                'max_duration': durations.max()
            }
    
    # Peak flux statistics
    peak_flux_stats = {}
    if peak_flux_column and peak_flux_column in flare_data.columns:
        peak_fluxes = flare_data[peak_flux_column].dropna()
        if len(peak_fluxes) > 0:
            peak_flux_stats = {
                'mean_peak_flux': peak_fluxes.mean(),
                'median_peak_flux': peak_fluxes.median(),
                'std_peak_flux': peak_fluxes.std(),
                'min_peak_flux': peak_fluxes.min(),
                'max_peak_flux': peak_fluxes.max()
            }
    
    # Temporal distribution analysis
    temporal_distribution = {}
    if time_column and time_column in flare_data.columns:
        times = pd.to_datetime(flare_data[time_column])
        temporal_distribution = {
            'time_span': (times.min(), times.max()),
            'flare_rate_per_day': len(times) / (times.max() - times.min()).days if len(times) > 1 else 0,
            'hourly_distribution': times.dt.hour.value_counts().to_dict(),
            'monthly_distribution': times.dt.month.value_counts().to_dict()
        }
    
    return FlareStatistics(
        n_flares=energy_stats['n_flares'],
        total_energy=energy_stats['total_energy'],
        mean_energy=energy_stats['mean_energy'],
        median_energy=energy_stats['median_energy'],
        energy_std=energy_stats['energy_std'],
        energy_range=energy_stats['energy_range'],
        duration_stats=duration_stats,
        peak_flux_stats=peak_flux_stats,
        temporal_distribution=temporal_distribution
    )


def advanced_power_law_analysis(data: Union[np.ndarray, List[float]], 
                               save_results: Optional[str] = None,
                               mcmc_samples: int = 5000,
                               use_mcmc: bool = False) -> Dict[str, Any]:
    """
    Perform advanced power-law analysis with MCMC sampling (if available).
    
    Parameters
    ----------
    data : array-like
        Data to analyze
    save_results : str, optional
        Path to save results
    mcmc_samples : int, default 5000
        Number of MCMC samples
    use_mcmc : bool, default False
        Whether to use MCMC sampling for parameter estimation
        
    Returns
    -------
    dict
        Comprehensive analysis results
    """
    logger.info("Starting advanced power-law analysis...")
    
    # Basic power-law fit
    basic_results = fit_power_law(data, plot=False)
    
    # Advanced analysis
    analysis_results = {
        'basic_fit': basic_results,
        'data_summary': {
            'n_points': len(data),
            'min_value': np.min(data),
            'max_value': np.max(data),
            'mean': np.mean(data),
            'std': np.std(data),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data)
        }
    }
    
    # MCMC analysis (if emcee is available and requested)
    if use_mcmc and HAS_EMCEE:
        try:
            mcmc_results = _mcmc_power_law_fit(data, basic_results.xmin, mcmc_samples)
            analysis_results['mcmc_fit'] = mcmc_results
        except Exception as e:
            logger.warning(f"MCMC analysis failed: {e}")
            analysis_results['mcmc_fit'] = None
    
    # Heavy-tail tests
    analysis_results['heavy_tail_tests'] = _heavy_tail_tests(data)
    
    # Scaling analysis
    analysis_results['scaling_analysis'] = _scaling_analysis(data)
    
    # Save results if requested
    if save_results:
        try:
            import json
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, (PowerLawResults, FlareStatistics)):
                    return obj.__dict__
                return obj
            
            json_results = json.dumps(analysis_results, default=convert_for_json, indent=2)
            with open(save_results, 'w') as f:
                f.write(json_results)
            logger.info(f"Results saved to {save_results}")
        except Exception as e:
            logger.warning(f"Failed to save results: {e}")
    
    return analysis_results


def _mcmc_power_law_fit(data: np.ndarray, xmin: float, n_samples: int) -> Dict[str, Any]:
    """Perform MCMC fitting for power-law parameters."""
    if not HAS_EMCEE:
        raise ImportError("emcee package required for MCMC analysis")
    
    # Filter data
    data_filtered = data[data >= xmin]
    n_data = len(data_filtered)
    
    def log_likelihood(alpha):
        if alpha <= 1.0 or alpha >= 10.0:  # Prior bounds
            return -np.inf
        return n_data * np.log(alpha - 1) - n_data * np.log(xmin) - alpha * np.sum(np.log(data_filtered / xmin))
    
    def log_posterior(params):
        alpha = params[0]
        return log_likelihood(alpha)
    
    # Initialize walkers
    n_walkers = 50
    n_dim = 1
    initial_alpha = 2.5  # Initial guess
    pos = initial_alpha + 0.1 * np.random.randn(n_walkers, n_dim)
    
    # Run MCMC
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_posterior)
    sampler.run_mcmc(pos, n_samples, progress=True)
    
    # Extract results
    samples = sampler.get_chain(discard=1000, flat=True)
    alpha_samples = samples[:, 0]
    
    return {
        'alpha_mean': np.mean(alpha_samples),
        'alpha_std': np.std(alpha_samples),
        'alpha_median': np.median(alpha_samples),
        'alpha_percentiles': np.percentile(alpha_samples, [2.5, 16, 50, 84, 97.5]),
        'samples': alpha_samples,
        'acceptance_fraction': np.mean(sampler.acceptance_fraction)
    }


def _heavy_tail_tests(data: np.ndarray) -> Dict[str, float]:
    """Perform tests for heavy-tail behavior."""
    tests = {}
    
    try:
        # Hill estimator for tail index
        data_sorted = np.sort(data)[::-1]  # Descending order
        n = len(data_sorted)
        k = min(n // 4, 100)  # Number of upper order statistics
        
        hill_estimates = []
        for i in range(1, k):
            hill_est = np.mean(np.log(data_sorted[:i] / data_sorted[i]))
            hill_estimates.append(hill_est)
        
        tests['hill_estimator'] = np.mean(hill_estimates) if hill_estimates else np.nan
        tests['hill_estimator_std'] = np.std(hill_estimates) if hill_estimates else np.nan
        
        # Moment ratio test
        if len(data) > 10:
            mean_val = np.mean(data)
            var_val = np.var(data)
            skew_val = stats.skew(data)
            kurt_val = stats.kurtosis(data)
            
            tests['moment_ratio'] = var_val / (mean_val ** 2) if mean_val > 0 else np.nan
            tests['skewness'] = skew_val
            tests['excess_kurtosis'] = kurt_val
        
    except Exception as e:
        logger.warning(f"Heavy-tail tests failed: {e}")
    
    return tests


def _scaling_analysis(data: np.ndarray) -> Dict[str, Any]:
    """Analyze scaling properties of the data."""
    analysis = {}
    
    try:
        # Log-log slope analysis across different ranges
        log_data = np.log10(data[data > 0])
        data_sorted = np.sort(data[data > 0])
        
        # Calculate slopes for different data ranges
        ranges = [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.5, 1.0)]
        slopes = []
        
        for start_frac, end_frac in ranges:
            start_idx = int(start_frac * len(data_sorted))
            end_idx = int(end_frac * len(data_sorted))
            
            if end_idx > start_idx + 2:
                x_range = data_sorted[start_idx:end_idx]
                y_range = np.arange(start_idx, end_idx) / len(data_sorted)
                
                # Fit in log-log space
                log_x = np.log10(x_range)
                log_y = np.log10(1 - y_range + 1e-10)  # Complementary CDF
                
                if len(log_x) > 2:
                    slope, _, _, _, _ = stats.linregress(log_x, log_y)
                    slopes.append(-slope)  # Convert to positive power-law exponent
        
        analysis['range_slopes'] = slopes
        analysis['slope_consistency'] = np.std(slopes) if slopes else np.nan
        
        # Scaling region detection
        if len(slopes) > 1:
            analysis['scaling_region_stable'] = np.std(slopes) < 0.2
        
    except Exception as e:
        logger.warning(f"Scaling analysis failed: {e}")
    
    return analysis

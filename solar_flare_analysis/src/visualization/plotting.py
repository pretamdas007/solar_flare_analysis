"""
Visualization utilities for solar flare analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.patches import Rectangle
import seaborn as sns

# Set default seaborn theme for all visualizations
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
sns.set_context("paper", rc={"figure.dpi": 300})


def plot_xrs_time_series(time_series, flux_column, title=None, figsize=(12, 6), 
                        log_scale=True, highlight_regions=None):
    """
    Plot a time series of GOES XRS flux data with professional seaborn aesthetics.
    
    Parameters
    ----------
    time_series : pandas.DataFrame
        DataFrame containing time index and flux values
    flux_column : str
        Name of the column containing flux data
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size
    log_scale : bool, optional
        If True, use logarithmic scale for y-axis
    highlight_regions : list, optional
        List of tuples with (start_time, end_time, label, color) to highlight
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot
    """
    # Set professional seaborn styling
    plt.style.use('seaborn-v0_8')
    sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the time series with seaborn styling
    sns.lineplot(x=time_series.index, y=time_series[flux_column], 
                ax=ax, color='steelblue', linewidth=2, label=flux_column)
      # Set labels and title with enhanced styling
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Flux (W/m²)', fontsize=12, fontweight='bold')
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'GOES XRS {flux_column} Time Series', fontsize=14, fontweight='bold')
    
    # Set log scale if requested
    if log_scale:
        ax.set_yscale('log')
    
    # Format x-axis for datetime
    if pd.api.types.is_datetime64_any_dtype(time_series.index):
        # Determine appropriate date formatter based on time span
        time_span = time_series.index[-1] - time_series.index[0]
        
        if time_span < pd.Timedelta(hours=24):
            # For less than a day, show hours and minutes
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif time_span < pd.Timedelta(days=14):
            # For less than two weeks, show day and hour
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H:%M'))
        else:
            # For longer periods, show date only
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        plt.xticks(rotation=45)
    
    # Highlight regions if specified
    if highlight_regions:
        for start, end, label, color in highlight_regions:
            ax.axvspan(start, end, alpha=0.3, color=color)
            # Add label in the middle of the span
            mid_time = start + (end - start) / 2
            if log_scale:
                y_pos = np.sqrt(ax.get_ylim()[0] * ax.get_ylim()[1])
            else:
                y_pos = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
            ax.text(mid_time, y_pos, label, 
                   ha='center', va='center', 
                   bbox=dict(facecolor='white', alpha=0.7))
          # Add enhanced flare classification reference lines
        flare_classes = {
            'A': (1e-8, 'darkgray'),
            'B': (1e-7, 'green'),
            'C': (1e-6, 'gold'),
            'M': (1e-5, 'orange'),
            'X': (1e-4, 'red'),
        }
        
        for cls, (level, color) in flare_classes.items():
            ax.axhline(y=level, color=color, linestyle='--', alpha=0.8, linewidth=1.5)
            ax.text(time_series.index[0], level, f'{cls}', 
                   va='center', ha='right', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.8, edgecolor=color))
    
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    return fig


def plot_detected_flares(time_series, flux_column, flares_df, 
                         background_column=None, figsize=(12, 8)):
    """
    Plot time series with highlighted detected flares.
    
    Parameters
    ----------
    time_series : pandas.DataFrame
        DataFrame containing time index and flux values
    flux_column : str
        Name of the column containing flux data
    flares_df : pandas.DataFrame
        DataFrame containing flare information (start_time, peak_time, end_time)
    background_column : str, optional
        Name of the column containing background flux
    figsize : tuple, optional
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the time series
    ax.plot(time_series.index, time_series[flux_column], 'b-', label='Flux')
    
    # Plot background if available
    if background_column and background_column in time_series.columns:
        ax.plot(time_series.index, time_series[background_column], 
               'g-', label='Background', alpha=0.7)
    
    # Set y-scale to log
    ax.set_yscale('log')
    
    # Highlight flare regions
    for i, flare in flares_df.iterrows():
        start = flare['start_time'] if 'start_time' in flare else flare['start_index']
        peak = flare['peak_time'] if 'peak_time' in flare else flare['peak_index']
        end = flare['end_time'] if 'end_time' in flare else flare['end_index']
        
        # Ensure we're working with datetime or indices consistently
        if isinstance(start, (int, np.integer)):
            start = time_series.index[start]
            peak = time_series.index[peak]
            end = time_series.index[end]
        
        # Highlight rise phase
        ax.axvspan(start, peak, alpha=0.2, color='red')
        
        # Highlight decay phase
        ax.axvspan(peak, end, alpha=0.2, color='orange')
        
        # Mark peak
        ax.plot(peak, time_series.loc[peak, flux_column], 'ro', markersize=8)
        
        # Add flare label
        label_y = time_series.loc[peak, flux_column]
        ax.text(peak, label_y * 1.1, f"Flare {i+1}", 
               ha='center', va='bottom', fontsize=8,
               bbox=dict(facecolor='white', alpha=0.7))
    
    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Flux (W/m²)')
    ax.set_title('Detected Solar Flares')
    
    # Format x-axis for datetime
    if pd.api.types.is_datetime64_any_dtype(time_series.index):
        time_span = time_series.index[-1] - time_series.index[0]
        
        if time_span < pd.Timedelta(hours=24):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif time_span < pd.Timedelta(days=14):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H:%M'))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
        plt.xticks(rotation=45)
    
    # Add GOES flare classification reference lines
    flare_classes = {
        'A': 1e-8,
        'B': 1e-7,
        'C': 1e-6,
        'M': 1e-5,
        'X': 1e-4,
    }
    
    for cls, level in flare_classes.items():
        ax.axhline(y=level, color='gray', linestyle='--', alpha=0.5)
        ax.text(time_series.index[0], level, f'{cls}', 
               va='center', ha='right', fontsize=8,
               bbox=dict(facecolor='white', alpha=0.7))
      # Enhanced grid and layout
    ax.legend(frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    
    return fig


def plot_flare_decomposition(original_flux, decomposed_flares, 
                             timestamps=None, figsize=(12, 10)):
    """
    Plot original flux and decomposed overlapping flares.
    
    Parameters
    ----------
    original_flux : array-like
        Original flux time series
    decomposed_flares : array-like
        Array with shape (time_points, n_flares) containing decomposed flares
    timestamps : array-like, optional
        Timestamps for the x-axis
    figsize : tuple, optional
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot
    """
    n_flares = decomposed_flares.shape[1]
    
    # Create x-axis values
    if timestamps is None:
        x = np.arange(len(original_flux))
    else:
        x = timestamps
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Plot original flux
    ax1 = fig.add_subplot(n_flares + 1, 1, 1)
    ax1.plot(x, original_flux, 'k-', label='Original Flux')
    
    # Calculate and plot reconstructed flux
    reconstructed = np.sum(decomposed_flares, axis=1)
    ax1.plot(x, reconstructed, 'r--', label='Reconstructed')
    
    ax1.set_title('Original and Reconstructed Flux')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylabel('Flux (W/m²)')
    ax1.set_yscale('log')
    
    if pd.api.types.is_datetime64_any_dtype(x):
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax1.get_xticklabels(), rotation=45)
    
    # Plot individual decomposed flares
    for i in range(n_flares):
        ax = fig.add_subplot(n_flares + 1, 1, i + 2, sharex=ax1)
        ax.plot(x, decomposed_flares[:, i], label=f'Flare {i+1}')
        ax.set_title(f'Decomposed Flare {i+1}')
        ax.grid(True)
        ax.set_ylabel('Flux (W/m²)')
        ax.set_yscale('log')
        
        if pd.api.types.is_datetime64_any_dtype(x):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(ax.get_xticklabels(), rotation=45)
    
    # Only show x-label on bottom subplot
    ax.set_xlabel('Time')
    
    plt.tight_layout()
    
    return fig


def plot_power_law_comparison(results_dict, labels=None, figsize=(10, 8)):
    """
    Plot power-law fit results for comparison.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with power-law fit results
    labels : list, optional
        Labels for the different results
    figsize : tuple, optional
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot
    """
    if labels is None:
        labels = list(results_dict.keys())
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    # Plot alpha values with error bars
    alphas = []
    errors = []
    for key in results_dict.keys():
        alphas.append(results_dict[key]['alpha'])
        errors.append(results_dict[key]['alpha_err'])
    
    y_pos = np.arange(len(labels))
    ax1.errorbar(alphas, y_pos, xerr=errors, fmt='o', capsize=5, color=colors)
    
    for i, (alpha, error) in enumerate(zip(alphas, errors)):
        ax1.text(alpha, y_pos[i], f'α = {alpha:.3f} ± {error:.3f}', 
                va='center', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7))
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('Power-law Exponent (α)')
    ax1.set_title('Comparison of Power-law Exponents')
    ax1.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Plot bootstrap distributions
    for i, key in enumerate(results_dict.keys()):
        if 'bootstrap_samples' in results_dict[key]:
            samples = results_dict[key]['bootstrap_samples']
            ax2.hist(samples, bins=30, alpha=0.5, label=labels[i], 
                    color=colors[i], density=True)
            ax2.axvline(results_dict[key]['alpha'], color=colors[i], 
                       linestyle='--', linewidth=2)
    
    ax2.set_xlabel('Power-law Exponent (α)')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Bootstrap Distributions')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig


def plot_flare_statistics(flares_df, figsize=(15, 10)):
    """
    Plot various statistics about detected flares with professional seaborn aesthetics.
    
    Parameters
    ----------
    flares_df : pandas.DataFrame
        DataFrame containing flare information
    figsize : tuple, optional
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot
    """
    # Set professional seaborn styling
    plt.style.use('seaborn-v0_8')
    sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
    
    fig = plt.figure(figsize=figsize)
    fig.suptitle('Comprehensive Solar Flare Statistics', fontsize=16, fontweight='bold', y=0.98)
    
    # Define subplot grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
      # Flare duration histogram with seaborn
    ax1 = fig.add_subplot(gs[0, 0])
    if 'duration' in flares_df.columns:
        if pd.api.types.is_timedelta64_dtype(flares_df['duration']):
            # Convert timedelta to minutes
            duration_min = flares_df['duration'].dt.total_seconds() / 60
            sns.histplot(duration_min, bins=20, alpha=0.8, ax=ax1, 
                        color='skyblue', kde=True, stat='density')
            ax1.set_xlabel('Duration (minutes)', fontsize=12, fontweight='bold')
        else:
            sns.histplot(flares_df['duration'], bins=20, alpha=0.8, ax=ax1, 
                        color='skyblue', kde=True, stat='density')
            ax1.set_xlabel('Duration', fontsize=12, fontweight='bold')
        
        ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax1.set_title('Flare Duration Distribution', fontsize=14, fontweight='bold')
      # Peak flux histogram with enhanced seaborn styling
    ax2 = fig.add_subplot(gs[0, 1])
    if 'peak_flux' in flares_df.columns:
        # Use log scale for peak flux
        log_flux = np.log10(flares_df['peak_flux'])
        sns.histplot(log_flux, bins=20, kde=True, ax=ax2, 
                    color='coral', alpha=0.8, stat='density')
        ax2.set_xlabel('Log₁₀(Peak Flux) [W/m²]', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax2.set_title('Peak Flux Distribution', fontsize=14, fontweight='bold')
        
        # Add median line
        median_log_flux = np.median(log_flux)
        ax2.axvline(median_log_flux, color='red', linestyle='--', linewidth=2,
                   label=f'Median: {10**median_log_flux:.2e} W/m²')
        ax2.legend()
      # Flare energy histogram with enhanced visualization
    ax3 = fig.add_subplot(gs[1, 0])
    if 'integrated_flux' in flares_df.columns:
        log_energy = np.log10(flares_df['integrated_flux'])
        sns.histplot(log_energy, bins=20, kde=True, ax=ax3, 
                    color='gold', alpha=0.8, stat='density')
        ax3.set_xlabel('Log₁₀(Integrated Flux) [Energy]', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax3.set_title('Flare Energy Distribution', fontsize=14, fontweight='bold')
        
        # Add mean and median lines
        mean_log = np.mean(log_energy)
        median_log = np.median(log_energy)
        ax3.axvline(mean_log, color='blue', linestyle='-', linewidth=2, 
                   label=f'Mean: {10**mean_log:.2e}')
        ax3.axvline(median_log, color='red', linestyle='--', linewidth=2,
                   label=f'Median: {10**median_log:.2e}')
        ax3.legend()
    
    # Rise vs decay time scatter plot
    ax4 = fig.add_subplot(gs[1, 1])
    if ('peak_time' in flares_df.columns and 
        'start_time' in flares_df.columns and 
        'end_time' in flares_df.columns):
        
        if pd.api.types.is_datetime64_any_dtype(flares_df['peak_time']):
            # Calculate rise and decay times in minutes
            rise_time = (flares_df['peak_time'] - flares_df['start_time']).dt.total_seconds() / 60
            decay_time = (flares_df['end_time'] - flares_df['peak_time']).dt.total_seconds() / 60
        else:
            rise_time = flares_df['peak_time'] - flares_df['start_time']
            decay_time = flares_df['end_time'] - flares_df['peak_time']
          # Create enhanced scatter plot
        scatter = ax4.scatter(rise_time, decay_time, alpha=0.7, s=60, 
                             c=range(len(rise_time)), cmap='viridis',
                             edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('Rise Time (minutes)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Decay Time (minutes)', fontsize=12, fontweight='bold')
        ax4.set_title('Rise vs. Decay Time Analysis', fontsize=14, fontweight='bold')
        
        # Add y=x reference line
        max_val = max(ax4.get_xlim()[1], ax4.get_ylim()[1])
        ax4.plot([0, max_val], [0, max_val], 'k--', alpha=0.7, linewidth=2,
                label='Equal Rise/Decay')
        ax4.legend()
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4, shrink=0.8)
        cbar.set_label('Flare Index', fontsize=10)
      # Flares over time with enhanced visualization
    ax5 = fig.add_subplot(gs[2, :])
    if 'peak_time' in flares_df.columns and 'peak_flux' in flares_df.columns:
        # Sort by peak time
        sorted_flares = flares_df.sort_values('peak_time')
        
        # Create enhanced scatter plot with color coding
        scatter = ax5.scatter(sorted_flares['peak_time'], sorted_flares['peak_flux'], 
                             alpha=0.7, s=50, c=np.log10(sorted_flares['peak_flux']),
                             cmap='plasma', edgecolors='black', linewidth=0.5)
        
        # Connect points with trend line
        sns.lineplot(x=sorted_flares['peak_time'], y=sorted_flares['peak_flux'], 
                    ax=ax5, color='gray', alpha=0.5, linewidth=1, estimator=None)
        
        ax5.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Peak Flux (W/m²)', fontsize=12, fontweight='bold')
        ax5.set_title('Flare Activity Timeline', fontsize=14, fontweight='bold')
        ax5.set_yscale('log')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax5, shrink=0.8)
        cbar.set_label('Log₁₀(Peak Flux)', fontsize=10)
        
        # Format x-axis for datetime
        if pd.api.types.is_datetime64_any_dtype(sorted_flares['peak_time']):
            time_span = sorted_flares['peak_time'].max() - sorted_flares['peak_time'].min()
            
            if time_span < pd.Timedelta(hours=24):
                ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            elif time_span < pd.Timedelta(days=14):
                ax5.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H:%M'))
            else:
                ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                
            plt.setp(ax5.get_xticklabels(), rotation=45)
        
        ax5.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    return fig


class FlareVisualization:
    """
    Comprehensive flare visualization class for the API
    """
    
    def __init__(self):
        """Initialize the visualization class"""
        self.default_figsize = (12, 8)
        
    def plot_flare_timeline(self, flares, nanoflares=None, figsize=None):
        """
        Plot a timeline of flares with different markers for nanoflares
        
        Parameters
        ----------
        flares : list
            List of flare dictionaries
        nanoflares : list, optional
            List of nanoflare dictionaries
        figsize : tuple, optional
            Figure size
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the plot
        """
        if figsize is None:
            figsize = self.default_figsize
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract data from flares
        timestamps = []
        intensities = []
        is_nano = []
        
        nano_timestamps = set()
        if nanoflares:
            nano_timestamps = {f.get('timestamp', '') for f in nanoflares}
        
        for flare in flares:
            ts = flare.get('timestamp', '')
            if ts:
                try:
                    # Parse ISO timestamp
                    dt = pd.to_datetime(ts)
                    timestamps.append(dt)
                    intensities.append(flare.get('intensity', 0))
                    is_nano.append(ts in nano_timestamps)
                except:
                    continue
        
        if not timestamps:
            # Create empty plot
            ax.text(0.5, 0.5, 'No valid timestamp data', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Flare Timeline (No Data)')
            return fig
        
        # Separate regular flares and nanoflares
        regular_times = [t for t, nano in zip(timestamps, is_nano) if not nano]
        regular_intensities = [i for i, nano in zip(intensities, is_nano) if not nano]
        nano_times = [t for t, nano in zip(timestamps, is_nano) if nano]
        nano_intensities = [i for i, nano in zip(intensities, is_nano) if nano]
        
        # Plot regular flares
        if regular_times:
            ax.scatter(regular_times, regular_intensities, 
                      c='blue', alpha=0.6, s=50, label='Regular Flares')
        
        # Plot nanoflares
        if nano_times:
            ax.scatter(nano_times, nano_intensities, 
                      c='red', alpha=0.8, s=30, marker='^', label='Nanoflares')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Intensity')
        ax.set_title('Solar Flare Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_energy_distribution(self, flares, bins=20, figsize=None):
        """
        Plot energy distribution histogram
        
        Parameters
        ----------
        flares : list
            List of flare dictionaries
        bins : int, optional
            Number of histogram bins
        figsize : tuple, optional
            Figure size
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the plot
        """
        if figsize is None:
            figsize = self.default_figsize
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract energies
        energies = [f.get('energy', 0) for f in flares if f.get('energy', 0) > 0]
        
        if not energies:
            ax.text(0.5, 0.5, 'No energy data available', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Energy Distribution (No Data)')
            return fig
        
        # Create histogram in log space
        log_energies = np.log10(energies)
        ax.hist(log_energies, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        
        ax.set_xlabel('log₁₀(Energy) [erg]')
        ax.set_ylabel('Count')
        ax.set_title('Flare Energy Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_power_law(self, flares, figsize=None):
        """
        Plot cumulative energy distribution to show power law
        
        Parameters
        ----------
        flares : list
            List of flare dictionaries
        figsize : tuple, optional
            Figure size
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the plot
        """
        if figsize is None:
            figsize = self.default_figsize
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract energies
        energies = [f.get('energy', 0) for f in flares if f.get('energy', 0) > 0]
        
        if not energies:
            ax.text(0.5, 0.5, 'No energy data available', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Power Law Distribution (No Data)')
            return fig
        
        # Sort energies
        sorted_energies = np.sort(energies)[::-1]  # Descending order
        
        # Create cumulative distribution
        counts = np.arange(1, len(sorted_energies) + 1)
        
        # Plot on log-log scale
        ax.loglog(sorted_energies, counts, 'bo-', alpha=0.6, markersize=4)
        
        # Try to fit a power law
        if len(sorted_energies) > 5:
            log_e = np.log10(sorted_energies)
            log_c = np.log10(counts)
            
            # Linear fit in log space
            coeffs = np.polyfit(log_e, log_c, 1)
            power_law_index = coeffs[0]
            
            # Plot fit line
            fit_line = 10**(coeffs[1]) * sorted_energies**coeffs[0]
            ax.loglog(sorted_energies, fit_line, 'r-', 
                     label=f'Power law fit: α = {power_law_index:.2f}')
            ax.legend()
        
        ax.set_xlabel('Energy [erg]')
        ax.set_ylabel('Cumulative Count')
        ax.set_title('Cumulative Energy Distribution (Power Law)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_summary_plot(self, flares, nanoflares=None, figsize=(15, 10)):
        """
        Create a comprehensive summary plot with multiple panels
        
        Parameters
        ----------
        flares : list
            List of flare dictionaries
        nanoflares : list, optional
            List of nanoflare dictionaries
        figsize : tuple, optional
            Figure size
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the multi-panel plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Panel 1: Timeline
        timestamps = []
        intensities = []
        nano_timestamps = set()
        if nanoflares:
            nano_timestamps = {f.get('timestamp', '') for f in nanoflares}
        
        for flare in flares:
            ts = flare.get('timestamp', '')
            if ts:
                try:
                    dt = pd.to_datetime(ts)
                    timestamps.append(dt)
                    intensities.append(flare.get('intensity', 0))
                except:
                    continue
        
        if timestamps:
            is_nano = [ts.isoformat() + 'Z' in nano_timestamps for ts in timestamps]
            regular_times = [t for t, nano in zip(timestamps, is_nano) if not nano]
            regular_intensities = [i for i, nano in zip(intensities, is_nano) if not nano]
            nano_times = [t for t, nano in zip(timestamps, is_nano) if nano]
            nano_intensities = [i for i, nano in zip(intensities, is_nano) if nano]
            
            if regular_times:
                ax1.scatter(regular_times, regular_intensities, 
                           c='blue', alpha=0.6, s=30, label='Regular Flares')
            if nano_times:
                ax1.scatter(nano_times, nano_intensities, 
                           c='red', alpha=0.8, s=20, marker='^', label='Nanoflares')
            ax1.legend()
        
        ax1.set_title('Flare Timeline')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Intensity')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Energy distribution
        energies = [f.get('energy', 0) for f in flares if f.get('energy', 0) > 0]
        if energies:
            log_energies = np.log10(energies)
            ax2.hist(log_energies, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        
        ax2.set_title('Energy Distribution')
        ax2.set_xlabel('log₁₀(Energy) [erg]')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Power law
        if energies:
            sorted_energies = np.sort(energies)[::-1]
            counts = np.arange(1, len(sorted_energies) + 1)
            ax3.loglog(sorted_energies, counts, 'bo-', alpha=0.6, markersize=3)
        
        ax3.set_title('Power Law Distribution')
        ax3.set_xlabel('Energy [erg]')
        ax3.set_ylabel('Cumulative Count')
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Statistics summary
        total_flares = len(flares)
        nano_count = len(nanoflares) if nanoflares else 0
        total_energy = sum(energies) if energies else 0
        avg_energy = np.mean(energies) if energies else 0
        
        stats_text = f"""
        Total Flares: {total_flares}
        Nanoflares: {nano_count}
        Nanoflare %: {nano_count/total_flares*100:.1f}%
        
        Total Energy: {total_energy:.2e} erg
        Average Energy: {avg_energy:.2e} erg
        """
        
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='center')
        ax4.set_title('Summary Statistics')
        ax4.axis('off')
        
        plt.tight_layout()
        return fig


# Additional utility classes for backward compatibility
class SolarFlareVisualizer(FlareVisualization):
    """Alias for FlareVisualization for backward compatibility"""
    pass

class FlareDataPlotter(FlareVisualization):
    """Alias for FlareVisualization for backward compatibility"""
    pass

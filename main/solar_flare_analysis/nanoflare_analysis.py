#!/usr/bin/env python3
"""
Solar Nanoflare Detection Analysis

This script analyzes alpha predictions to detect nanoflare activity using the power-law threshold method.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm, gaussian_kde
from pathlib import Path
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

class NanoflareAnalyzer:
    """
    Nanoflare detection based on alpha power-law analysis
    """
    def __init__(self, alpha_threshold=2.0, alpha_uncertainty=0.03):
        """
        Initialize nanoflare analyzer based on Parker's nanoflare hypothesis
        
        Parameters:
        -----------
        alpha_threshold : float
            Alpha threshold for nanoflare detection (default: 2.0 - Parker's criterion)
        alpha_uncertainty : float
            Uncertainty in alpha threshold (default: ±0.03)
        """
        self.alpha_threshold = alpha_threshold
        self.alpha_uncertainty = alpha_uncertainty
        print(f"Parker's Nanoflare threshold: α = {alpha_threshold:.3f} ± {alpha_uncertainty:.3f}")
        print(f"α > {alpha_threshold:.1f} indicates nanoflares can power solar coronal heating")
    
    def smooth_distribution(self, values, sigma=1.5):
        """Apply Gaussian smoothing to frequency distribution"""
        return gaussian_filter1d(values, sigma=sigma)
    
    def calculate_power_law_index(self, alpha_values, smooth=True, n_bins=15):
        """
        Calculate power-law index from alpha value distribution
        
        Parameters:
        -----------
        alpha_values : array-like
            Array of alpha predictions
        smooth : bool
            Whether to smooth the frequency distribution
        n_bins : int
            Number of bins for frequency distribution
            
        Returns:
        --------
        dict
            Power-law analysis results
        """
        # Create histogram of alpha values
        counts, bin_edges = np.histogram(alpha_values, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Remove zero counts
        mask = counts > 0
        bin_centers_clean = bin_centers[mask]
        counts_clean = counts[mask]
        
        if len(bin_centers_clean) < 3:
            return {
                'alpha': np.nan,
                'alpha_error': np.nan,
                'r_squared': np.nan,
                'fit_quality': 'insufficient_data',
                'n_points': len(bin_centers_clean)
            }
        
        # Smooth if requested
        if smooth and len(counts_clean) > 5:
            counts_clean = self.smooth_distribution(counts_clean)
        
        # Power-law fitting in log-log space
        log_alpha = np.log10(bin_centers_clean)
        log_counts = np.log10(counts_clean)
        
        try:
            # Linear regression: log(N) = -β * log(α) + const
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_alpha, log_counts)
            
            # Power-law index is negative of slope
            beta = -slope  # This is our power-law index
            beta_error = std_err
            r_squared = r_value**2
            
            # Determine fit quality
            if r_squared > 0.8:
                fit_quality = 'excellent'
            elif r_squared > 0.6:
                fit_quality = 'good'
            elif r_squared > 0.4:
                fit_quality = 'fair'
            else:
                fit_quality = 'poor'
            
            return {
                'alpha': beta,  # This is our measured power-law index
                'alpha_error': beta_error,
                'r_squared': r_squared,
                'p_value': p_value,
                'fit_quality': fit_quality,
                'slope': slope,
                'intercept': intercept,
                'n_points': len(bin_centers_clean),
                'bin_centers': bin_centers_clean,
                'counts': counts_clean,
                'log_alpha': log_alpha,
                'log_counts': log_counts
            }
            
        except Exception as e:
            return {
                'alpha': np.nan,
                'alpha_error': np.nan,
                'r_squared': np.nan,
                'fit_quality': f'error: {str(e)}',
                'n_points': len(bin_centers_clean)
            }
    
    def detect_nanoflares(self, power_law_result, confidence_level=0.95):
        """
        Detect nanoflares based on alpha threshold criteria
        
        Parameters:
        -----------
        power_law_result : dict
            Result from calculate_power_law_index
        confidence_level : float
            Confidence level for detection
            
        Returns:
        --------
        dict
            Nanoflare detection results
        """
        measured_alpha = power_law_result.get('alpha', np.nan)
        alpha_error = power_law_result.get('alpha_error', np.nan)
        
        if np.isnan(measured_alpha):
            return {
                'is_nanoflare': False,
                'detection_confidence': 0.0,
                'reason': 'alpha_calculation_failed',
                'measured_alpha': measured_alpha,
                'threshold_comparison': 'undefined'
            }
        
        # Check if measured alpha is greater than threshold
        alpha_above_threshold = measured_alpha > self.alpha_threshold
        
        # Calculate confidence considering uncertainties
        if not np.isnan(alpha_error):
            # Combined uncertainty
            combined_uncertainty = np.sqrt(alpha_error**2 + self.alpha_uncertainty**2)
            
            # Calculate z-score for statistical significance
            z_score = (measured_alpha - self.alpha_threshold) / combined_uncertainty
            
            # Convert to confidence using normal distribution
            if alpha_above_threshold:
                detection_confidence = norm.cdf(z_score)
            else:
                detection_confidence = norm.cdf(-z_score)
        else:
            # Simple threshold comparison without error consideration
            detection_confidence = 0.8 if alpha_above_threshold else 0.2
        
        # Determine if this qualifies as nanoflare detection
        fit_quality_good = power_law_result.get('r_squared', 0) > 0.5
        is_nanoflare = (alpha_above_threshold and 
                       detection_confidence >= confidence_level and
                       fit_quality_good)
        
        # Prepare detailed comparison
        threshold_comparison = {
            'measured_alpha': measured_alpha,
            'alpha_error': alpha_error,
            'threshold': self.alpha_threshold,
            'threshold_uncertainty': self.alpha_uncertainty,
            'difference': measured_alpha - self.alpha_threshold,
            'sigma_difference': (measured_alpha - self.alpha_threshold) / combined_uncertainty if not np.isnan(alpha_error) else np.nan
        }
          # Determine reason
        if not fit_quality_good:
            reason = 'poor_power_law_fit'
        elif not alpha_above_threshold:
            reason = 'alpha_below_parker_threshold'
        elif detection_confidence < confidence_level:
            reason = 'low_confidence'
        else:
            reason = 'nanoflares_detected_coronal_heating_capable'
        
        return {
            'is_nanoflare': is_nanoflare,
            'detection_confidence': detection_confidence,
            'alpha_above_threshold': alpha_above_threshold,
            'reason': reason,
            'threshold_comparison': threshold_comparison,
            'fit_quality': power_law_result.get('fit_quality', 'unknown'),
            'z_score': z_score if not np.isnan(alpha_error) else np.nan
        }
    
    def analyze_alpha_predictions(self, alpha_values):
        """
        Complete nanoflare analysis from alpha predictions
        
        Parameters:
        -----------
        alpha_values : array-like
            Array of alpha predictions
            
        Returns:
        --------
        dict
            Complete analysis results
        """
        print(f"Analyzing {len(alpha_values)} alpha predictions...")
        
        # Calculate power-law index from distribution
        power_law_result = self.calculate_power_law_index(alpha_values)
        
        # Detect nanoflares
        nanoflare_result = self.detect_nanoflares(power_law_result)
          # Calculate additional statistics
        alpha_stats = {
            'mean': np.mean(alpha_values),
            'median': np.median(alpha_values),
            'std': np.std(alpha_values),
            'min': np.min(alpha_values),
            'max': np.max(alpha_values),
            'count': len(alpha_values)
        }
        return {
            'alpha_statistics': alpha_stats,
            'power_law_analysis': power_law_result,
            'nanoflare_detection': nanoflare_result,
            'analysis_successful': True
        }      
    def plot_analysis(self, analysis_result, alpha_values, output_path=None):
        """
        Create ultra-modern, publication-ready visualization of Parker's nanoflare analysis
        with sophisticated design principles and accessibility features
        """
        # Set ultra-modern seaborn style with custom context
        sns.set_theme(
            style="ticks",
            palette="Set2", 
            context="paper",
            rc={
                "axes.spines.right": False,
                "axes.spines.top": False,
                "axes.grid": True,
                "grid.alpha": 0.15,
                "grid.linewidth": 0.5,
                "figure.facecolor": "white",
                "axes.facecolor": "#fafafa"
            }
        )
        
        # Professional, accessible color palette - colorblind friendly
        colors = {
            'primary': '#0173b2',       # Royal blue (accessible)
            'secondary': '#de8f05',     # Amber (high contrast)
            'accent': '#029e73',        # Teal green
            'success': '#cc78bc',       # Soft magenta  
            'info': '#ca9161',          # Warm brown
            'threshold': '#d55e00',     # Vermillion (high visibility)
            'mean': '#56b4e9',          # Sky blue
            'fit': '#009e73',           # Bluish green
            'kde': '#8b008b',           # Dark magenta
            'confidence': '#999999',    # Medium gray
            'background': '#fafafa',    # Light gray background
            'text': '#2c2c2c',          # Dark gray text
            'highlight': '#ffeb3b'      # Yellow highlight
        }
        
        # Ultra-modern typography with enhanced readability
        plt.rcParams.update({
            'font.family': ['Source Sans Pro', 'Helvetica Neue', 'Arial', 'sans-serif'],
            'font.size': 11,
            'font.weight': 'normal',
            'axes.labelsize': 13,
            'axes.titlesize': 15,
            'axes.titleweight': 'bold',
            'axes.labelweight': 'medium',
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 9,
            'figure.titlesize': 18,
            'figure.titleweight': 'bold',
            'axes.linewidth': 1.2,
            'grid.alpha': 0.15,
            'grid.linewidth': 0.6,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'axes.axisbelow': True,
            'axes.edgecolor': '#cccccc',
            'text.color': colors['text'],
            'axes.labelcolor': colors['text'],
            'xtick.color': colors['text'],
            'ytick.color': colors['text']
        })        
        # Create ultra-modern figure with sophisticated layout
        fig = plt.figure(figsize=(20, 16), facecolor='white')
        fig.patch.set_facecolor('white')
        
        # Advanced grid layout with golden ratio proportions and better spacing
        gs = fig.add_gridspec(
            3, 3, 
            height_ratios=[1.2, 1.2, 0.5], 
            width_ratios=[1.0, 1.0, 1.0],
            hspace=0.4, 
            wspace=0.35,
            left=0.08, 
            right=0.95, 
            top=0.90, 
            bottom=0.12
        )
        
        alpha_stats = analysis_result['alpha_statistics']
        power_law = analysis_result['power_law_analysis']
        nanoflare = analysis_result['nanoflare_detection']
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # PANEL 1: Ultra-modern Alpha Distribution with Advanced Statistical Overlays
        # ═══════════════════════════════════════════════════════════════════════════════
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(colors['background'])
        
        # Modern histogram with refined bins and subtle transparency
        n_bins = max(20, min(50, len(alpha_values) // 10))
        n, bins, patches = ax1.hist(
            alpha_values, 
            bins=n_bins, 
            alpha=0.7, 
            color=colors['primary'], 
            edgecolor='white', 
            linewidth=0.8, 
            density=True,
            label='Alpha Distribution'
        )
        
        # Sophisticated color gradient for histogram bars
        cm = plt.cm.Blues
        for i, p in enumerate(patches):
            intensity = 0.4 + 0.6 * (i / len(patches))
            p.set_facecolor(cm(intensity))
            p.set_alpha(0.8)
        
        # Advanced KDE with sophisticated styling
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(alpha_values)
        x_range = np.linspace(alpha_values.min() - 0.1, alpha_values.max() + 0.1, 400)
        kde_vals = kde(x_range)
        
        # KDE with elegant fill and refined line
        ax1.fill_between(x_range, kde_vals, alpha=0.25, color=colors['kde'], 
                        label='Kernel Density', zorder=5)
        ax1.plot(x_range, kde_vals, color=colors['kde'], linewidth=2.5, alpha=0.9, zorder=6)
        
        # Sophisticated threshold and statistical lines with enhanced styling
        ax1.axvline(
            self.alpha_threshold, 
            color=colors['threshold'], 
            linestyle='--', 
            linewidth=3, 
            alpha=0.9, 
            label=f'Parker Threshold (α = {self.alpha_threshold:.1f})', 
            zorder=10
        )
        
        ax1.axvline(
            alpha_stats['mean'], 
            color=colors['mean'], 
            linestyle='-', 
            linewidth=2.5, 
            alpha=0.8, 
            label=f'Mean α = {alpha_stats["mean"]:.3f}', 
            zorder=8
        )
        
        # Elegant percentile shading
        p25, p75 = np.percentile(alpha_values, [25, 75])
        ax1.axvspan(p25, p75, alpha=0.12, color=colors['info'], 
                   label=f'IQR: [{p25:.3f}, {p75:.3f}]', zorder=2)
        
        # Refined axis styling and labels
        ax1.set_xlabel('Alpha Values (α)', fontweight='medium', color=colors['text'])
        ax1.set_ylabel('Probability Density', fontweight='medium', color=colors['text'])
        ax1.set_title('Alpha Distribution Analysis\nwith Statistical Overlays', 
                     fontweight='bold', pad=20, color=colors['text'])
        
        # Modern legend with subtle styling
        legend1 = ax1.legend(
            frameon=True, 
            fancybox=False, 
            shadow=False, 
            loc='upper right', 
            fontsize=9, 
            framealpha=0.95,
            edgecolor='#dddddd',
            facecolor='white'
        )
        legend1.get_frame().set_linewidth(0.8)
        
        # Subtle grid enhancement
        ax1.grid(True, alpha=0.15, linestyle='-', linewidth=0.6, color='#cccccc')
        sns.despine(ax=ax1, top=True, right=True, left=False, bottom=False)        
        # ═══════════════════════════════════════════════════════════════════════════════
        # PANEL 2: Ultra-modern Power-law Analysis with Advanced Statistical Modeling
        # ═══════════════════════════════════════════════════════════════════════════════
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(colors['background'])
        
        if 'bin_centers' in power_law and 'counts' in power_law:
            bin_centers = power_law['bin_centers']
            counts = power_law['counts']
            log_alpha = power_law['log_alpha']
            log_counts = power_law['log_counts']
            
            # Sophisticated scatter plot with size and transparency mapping
            sizes = 40 + (counts - counts.min()) / (counts.max() - counts.min() + 1e-10) * 120
            
            # Create scatter with elegant color mapping
            scatter = ax2.scatter(
                log_alpha, log_counts, 
                alpha=0.8, 
                s=sizes, 
                c=counts, 
                cmap='viridis', 
                edgecolors='white', 
                linewidths=1.5,
                label='Observed Data',
                zorder=8
            )
            
            # Refined colorbar with better positioning
            cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8, aspect=25, pad=0.02)
            cbar.set_label('Frequency Count', fontweight='medium', fontsize=11, color=colors['text'])
            cbar.ax.tick_params(labelsize=9, colors=colors['text'])
            cbar.outline.set_edgecolor('#dddddd')
            cbar.outline.set_linewidth(0.8)
            
            # Advanced fit line with sophisticated confidence intervals
            if not np.isnan(power_law.get('alpha', np.nan)):
                slope = power_law['slope']
                intercept = power_law['intercept']
                fit_line = slope * log_alpha + intercept
                
                # Enhanced confidence calculation with bootstrap-like approach
                residuals = log_counts - fit_line
                mse = np.mean(residuals**2)
                confidence_95 = 1.96 * np.sqrt(mse)
                confidence_68 = 1.0 * np.sqrt(mse)
                
                # Multiple confidence bands with gradient effect
                ax2.fill_between(
                    log_alpha, 
                    fit_line - confidence_95, 
                    fit_line + confidence_95, 
                    alpha=0.15, 
                    color=colors['threshold'], 
                    label='95% Confidence',
                    zorder=3
                )
                ax2.fill_between(
                    log_alpha, 
                    fit_line - confidence_68, 
                    fit_line + confidence_68, 
                    alpha=0.25, 
                    color=colors['threshold'], 
                    label='68% Confidence',
                    zorder=4
                )
                
                # Elegant fit line
                ax2.plot(
                    log_alpha, fit_line, 
                    color=colors['threshold'], 
                    linewidth=3.5, 
                    label=f'β = {power_law["alpha"]:.3f} ± {power_law.get("alpha_error", 0):.3f}',
                    alpha=0.9, 
                    zorder=10
                )
                
                # Refined legend
                legend2 = ax2.legend(
                    frameon=True, 
                    fancybox=False, 
                    shadow=False, 
                    loc='upper right', 
                    fontsize=9, 
                    framealpha=0.95,
                    edgecolor='#dddddd',
                    facecolor='white'
                )
                legend2.get_frame().set_linewidth(0.8)
            
            ax2.set_xlabel('log₁₀(Alpha)', fontweight='medium', color=colors['text'])
            ax2.set_ylabel('log₁₀(Frequency)', fontweight='medium', color=colors['text'])
            ax2.set_title(f'Power-Law Fit Analysis\n(R² = {power_law.get("r_squared", 0):.3f})', 
                         fontweight='bold', pad=20, color=colors['text'])
        else:
            # Elegant error message
            ax2.text(
                0.5, 0.5, 
                'Power-law Fit Unavailable\n\nInsufficient Data Points\nfor Reliable Analysis', 
                ha='center', va='center', 
                transform=ax2.transAxes, 
                fontsize=12, fontweight='medium',
                bbox=dict(
                    boxstyle='round,pad=1.0', 
                    facecolor='#f5f5f5', 
                    alpha=0.9, 
                    edgecolor='#cccccc', 
                    linewidth=1
                ),
                color=colors['text']
            )
            ax2.set_title('Power-Law Analysis', fontweight='bold', pad=20, color=colors['text'])
        
        ax2.grid(True, alpha=0.15, linestyle='-', linewidth=0.6, color='#cccccc')
        sns.despine(ax=ax2, top=True, right=True)        
        # ═══════════════════════════════════════════════════════════════════════════════
        # PANEL 3: Modern Distribution Shape Analysis with Seaborn Integration
        # ═══════════════════════════════════════════════════════════════════════════════
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_facecolor(colors['background'])
        
        # Create sophisticated violin plot using seaborn
        violin_data = pd.DataFrame({'Alpha': alpha_values, 'Category': 'Distribution'})
        
        # Modern violin plot with refined styling
        violin = sns.violinplot(
            data=violin_data, 
            y='Alpha', 
            x='Category',
            ax=ax3,
            color=colors['primary'],
            alpha=0.7,
            inner='quart',
            linewidth=1.5
        )
        
        # Enhance violin appearance
        for collection in violin.collections:
            collection.set_alpha(0.7)
            collection.set_edgecolor('white')
            collection.set_linewidth(1.2)
        
        # Overlay refined box plot
        box_data = [alpha_values]
        bp = ax3.boxplot(
            box_data, 
            positions=[0], 
            widths=0.35, 
            patch_artist=True,
            boxprops=dict(
                facecolor=colors['accent'], 
                alpha=0.6, 
                edgecolor='white', 
                linewidth=1.5
            ),
            medianprops=dict(color='white', linewidth=2.5),
            whiskerprops=dict(color=colors['text'], linewidth=1.5),
            capprops=dict(color=colors['text'], linewidth=1.5),
            flierprops=dict(
                marker='o', 
                markerfacecolor=colors['threshold'], 
                markersize=6, 
                alpha=0.8,
                markeredgecolor='white',
                markeredgewidth=0.5
            ),
            showfliers=True
        )
        
        # Sophisticated threshold line with elegant annotation
        threshold_line = ax3.axhline(
            self.alpha_threshold, 
            color=colors['threshold'], 
            linestyle='--', 
            linewidth=3, 
            alpha=0.9, 
            zorder=10
        )
        
        # Modern threshold annotation with callout
        ax3.annotate(
            f'Parker Threshold\nα = {self.alpha_threshold:.1f}', 
            xy=(0.02, self.alpha_threshold), 
            xytext=(0.25, self.alpha_threshold + 0.2),
            fontsize=10, fontweight='medium', ha='left',
            bbox=dict(
                boxstyle='round,pad=0.4', 
                facecolor=colors['threshold'], 
                alpha=0.8, 
                edgecolor='white', 
                linewidth=1
            ),
            arrowprops=dict(
                arrowstyle='->', 
                lw=1.5, 
                color=colors['threshold'],
                alpha=0.8
            ),
            color='white'
        )
        
        ax3.set_ylabel('Alpha Values (α)', fontweight='medium', color=colors['text'])
        ax3.set_xlabel('')
        ax3.set_title('Distribution Shape Analysis\n(Parker\'s Nanoflare Criterion)', 
                     fontweight='bold', pad=20, color=colors['text'])
        ax3.set_xlim(-0.4, 0.4)
        ax3.set_xticks([])
        ax3.grid(True, alpha=0.15, linestyle='-', linewidth=0.6, color='#cccccc')
        sns.despine(ax=ax3, top=True, right=True, bottom=True)        
        # ═══════════════════════════════════════════════════════════════════════════════
        # PANEL 4: Threshold Comparison with Publication-Quality Bar Chart
        # ═══════════════════════════════════════════════════════════════════════════════
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.set_facecolor(colors['background'])
        
        measured_alpha = power_law.get('alpha', np.nan)
        alpha_err = power_law.get('alpha_error', 0)
        
        if not np.isnan(measured_alpha):
            categories = ['Parker\nThreshold\n(α = 2.0)', 'Measured β\n(This Study)']
            values = [self.alpha_threshold, measured_alpha]
            errors = [self.alpha_uncertainty, alpha_err]
            
            # Sophisticated color selection based on heating capability
            bar_colors = [
                colors['threshold'], 
                colors['success'] if nanoflare['is_nanoflare'] else colors['primary']
            ]
            
            # Create refined bar chart
            bars = ax4.bar(
                categories, values, 
                yerr=errors, 
                capsize=8, 
                color=bar_colors, 
                alpha=0.8, 
                edgecolor='white', 
                linewidth=1.5,
                error_kw={
                    'linewidth': 2.5, 
                    'alpha': 0.9, 
                    'capthick': 2,
                    'color': colors['text']
                }
            )
            
            # Enhanced significance analysis and annotation
            if not np.isnan(nanoflare.get('z_score', np.nan)):
                z_score = nanoflare['z_score']
                if abs(z_score) > 1.96:  # 95% confidence
                    significance = f'{abs(z_score):.1f}σ'
                    coronal_status = ("🔥 CORONAL HEATING\nCAPABLE" 
                                    if measured_alpha > self.alpha_threshold 
                                    else "❄️ INSUFFICIENT FOR\nCORONAL HEATING")
                    
                    # Sophisticated status annotation
                    bbox_color = ('#e8f5e8' if measured_alpha > self.alpha_threshold 
                                else '#ffebee')
                    text_color = ('#2e7d32' if measured_alpha > self.alpha_threshold 
                                else '#c62828')
                    
                    ax4.annotate(
                        f'Significance: {significance}\n{coronal_status}', 
                        xy=(0.5, max(values) + max(errors) * 1.2),
                        xytext=(0.5, max(values) + max(errors) * 1.8),
                        ha='center', va='center', 
                        fontsize=10, fontweight='medium',
                        bbox=dict(
                            boxstyle='round,pad=0.6', 
                            facecolor=bbox_color, 
                            alpha=0.9,
                            edgecolor=text_color, 
                            linewidth=1.5
                        ),
                        arrowprops=dict(
                            arrowstyle='->', 
                            lw=2, 
                            color=text_color,
                            alpha=0.8
                        ),
                        color=text_color
                    )
            
            # Refined value labels
            for i, (bar, val, err) in enumerate(zip(bars, values, errors)):
                height = bar.get_height()
                ax4.text(
                    bar.get_x() + bar.get_width()/2., 
                    height + err + 0.03,
                    f'{val:.3f}±{err:.3f}', 
                    ha='center', va='bottom', 
                    fontweight='medium', fontsize=10,
                    bbox=dict(
                        boxstyle='round,pad=0.3', 
                        facecolor='white', 
                        alpha=0.95, 
                        edgecolor='#cccccc', 
                        linewidth=0.8
                    ),
                    color=colors['text']
                )
            
            ax4.set_ylabel('Power-Law Index (β)', fontweight='medium', color=colors['text'])
            ax4.set_title('Parker\'s Nanoflare Threshold Analysis\n(Coronal Heating Capability)', 
                         fontweight='bold', pad=20, color=colors['text'])
        else:
            # Elegant unavailable message
            ax4.text(
                0.5, 0.5, 
                'Threshold Comparison\nUnavailable\n\nInsufficient Data\nfor Power-Law Fit', 
                ha='center', va='center', 
                transform=ax4.transAxes,
                fontsize=11, fontweight='medium',
                bbox=dict(
                    boxstyle='round,pad=0.8', 
                    facecolor='#f5f5f5', 
                    alpha=0.9, 
                    edgecolor='#cccccc', 
                    linewidth=1
                ),
                color=colors['text']
            )
            ax4.set_title('Threshold Comparison', fontweight='bold', pad=20, color=colors['text'])
        
        ax4.grid(True, alpha=0.15, linestyle='-', linewidth=0.6, color='#cccccc')
        sns.despine(ax=ax4, top=True, right=True)        
        # ═══════════════════════════════════════════════════════════════════════════════
        # PANEL 5: Advanced Confidence Interval Analysis 
        # ═══════════════════════════════════════════════════════════════════════════════
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set_facecolor(colors['background'])
        
        if not np.isnan(measured_alpha) and not np.isnan(alpha_err):
            # Sophisticated confidence interval visualization
            confidence_levels = [0.68, 0.95, 0.99]  # 1σ, 2σ, 3σ
            conf_colors = ['#e3f2fd', '#fff8e1', '#fce4ec']
            edge_colors = ['#1565c0', '#f57c00', '#ad1457']
            
            # Create elegant confidence regions
            for i, (conf, fill_color, edge_color) in enumerate(zip(confidence_levels, conf_colors, edge_colors)):
                z_val = norm.ppf((1 + conf) / 2)
                lower = measured_alpha - z_val * alpha_err
                upper = measured_alpha + z_val * alpha_err
                
                # Refined confidence region
                ax5.fill_between(
                    [i-0.3, i+0.3], [lower, lower], [upper, upper], 
                    color=fill_color, alpha=0.8, 
                    edgecolor=edge_color, linewidth=1.5,
                    label=f'{conf:.0%} CI: [{lower:.3f}, {upper:.3f}]'
                )
                
                # Elegant confidence level labels
                ax5.text(
                    i, (lower + upper) / 2, f'{conf:.0%}', 
                    ha='center', va='center',
                    fontweight='bold', fontsize=11, 
                    color=edge_color
                )
            
            # Sophisticated threshold and measurement lines
            threshold_line = ax5.axhline(
                self.alpha_threshold, 
                color=colors['threshold'], 
                linestyle='--', 
                linewidth=3, 
                alpha=0.9, 
                zorder=10,
                label=f'Parker Threshold: {self.alpha_threshold:.1f}'
            )
            
            measured_line = ax5.plot(
                [0, 1, 2], [measured_alpha]*3, 'o-', 
                color=colors['success'], 
                markersize=10, 
                linewidth=3, 
                markeredgecolor='white', 
                markeredgewidth=1.5,
                label=f'Measured: {measured_alpha:.3f}', 
                zorder=11
            )
            
            # Enhanced status annotation
            heating_status = ("🔥 CORONAL HEATING CAPABLE" 
                            if measured_alpha > self.alpha_threshold 
                            else "❄️ INSUFFICIENT FOR HEATING")
            status_color = ('#e8f5e8' if measured_alpha > self.alpha_threshold 
                          else '#ffebee')
            text_color = ('#2e7d32' if measured_alpha > self.alpha_threshold 
                        else '#c62828')
            
            ax5.text(
                1, measured_alpha + (max(confidence_levels) * alpha_err * 0.25), 
                heating_status, ha='center', va='bottom',
                fontweight='medium', fontsize=10, 
                bbox=dict(
                    boxstyle='round,pad=0.4', 
                    facecolor=status_color, 
                    alpha=0.9, 
                    edgecolor=text_color, 
                    linewidth=1.2
                ),
                color=text_color
            )
            
            ax5.set_xticks([0, 1, 2])
            ax5.set_xticklabels(['68%', '95%', '99%'], fontweight='medium')
            ax5.set_xlabel('Confidence Level', fontweight='medium', color=colors['text'])
            ax5.set_ylabel('Power-Law Index (β)', fontweight='medium', color=colors['text'])
            ax5.set_title('Confidence Interval Analysis\n(Statistical Uncertainty)', 
                         fontweight='bold', pad=20, color=colors['text'])
            
            # Refined legend
            legend5 = ax5.legend(
                frameon=True, 
                fancybox=False, 
                shadow=False, 
                loc='upper left', 
                fontsize=8, 
                framealpha=0.95,
                edgecolor='#dddddd',
                facecolor='white'
            )
            legend5.get_frame().set_linewidth(0.8)
        else:
            # Elegant unavailable message
            ax5.text(
                0.5, 0.5, 
                'Confidence Analysis\nUnavailable\n\nNo Error Estimates\nProvided', 
                ha='center', va='center', 
                transform=ax5.transAxes,
                fontsize=11, fontweight='medium',
                bbox=dict(
                    boxstyle='round,pad=0.8', 
                    facecolor='#f5f5f5', 
                    alpha=0.9, 
                    edgecolor='#cccccc', 
                    linewidth=1
                ),
                color=colors['text']
            )
            ax5.set_title('Confidence Analysis', fontweight='bold', pad=20, color=colors['text'])
        
        ax5.grid(True, alpha=0.15, linestyle='-', linewidth=0.6, color='#cccccc')
        sns.despine(ax=ax5, top=True, right=True)        
        # ═══════════════════════════════════════════════════════════════════════════════
        # PANEL 6: Advanced Residuals Analysis with Statistical Diagnostics
        # ═══════════════════════════════════════════════════════════════════════════════
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_facecolor(colors['background'])
        
        if ('log_alpha' in power_law and 'log_counts' in power_law 
            and not np.isnan(measured_alpha)):
            
            log_alpha = power_law['log_alpha']
            log_counts = power_law['log_counts']
            slope = power_law['slope']
            intercept = power_law['intercept']
            
            # Calculate residuals and diagnostics
            predicted = slope * log_alpha + intercept
            residuals = log_counts - predicted
            
            # Sophisticated scatter plot with diagnostic coloring
            abs_residuals = np.abs(residuals)
            max_abs_residual = np.max(abs_residuals)
            
            scatter = ax6.scatter(
                predicted, residuals, 
                alpha=0.8, s=80, 
                c=abs_residuals,
                cmap='RdYlBu_r', 
                edgecolors='white', 
                linewidths=1.2,
                label='Residuals',
                vmin=0, vmax=max_abs_residual
            )
            
            # Refined colorbar
            cbar6 = plt.colorbar(scatter, ax=ax6, shrink=0.8, aspect=25, pad=0.02)
            cbar6.set_label('|Residuals|', fontweight='medium', fontsize=11, color=colors['text'])
            cbar6.ax.tick_params(labelsize=9, colors=colors['text'])
            cbar6.outline.set_edgecolor('#dddddd')
            cbar6.outline.set_linewidth(0.8)
            
            # Sophisticated zero line
            ax6.axhline(
                0, 
                color=colors['threshold'], 
                linestyle='--', 
                linewidth=2.5, 
                alpha=0.9,
                label='Zero Residuals', 
                zorder=10
            )
            
            # Advanced trend analysis
            if len(predicted) > 3:
                z = np.polyfit(predicted, residuals, 1)
                p = np.poly1d(z)
                trend_line = p(predicted)
                
                ax6.plot(
                    predicted, trend_line, 
                    color=colors['accent'], 
                    linewidth=2.5, 
                    alpha=0.9, 
                    label=f'Trend: slope={z[0]:.3f}', 
                    zorder=11
                )
                
                # Confidence band for trend
                residual_std = np.std(residuals - trend_line)
                ax6.fill_between(
                    predicted, 
                    trend_line - 1.96*residual_std, 
                    trend_line + 1.96*residual_std, 
                    alpha=0.15, 
                    color=colors['accent'], 
                    label='95% Confidence'
                )
            
            # Enhanced statistical diagnostics
            rmse = np.sqrt(np.mean(residuals**2))
            mae = np.mean(abs_residuals)
            mean_residual = np.mean(residuals)
            
            stats_text = (f'RMSE: {rmse:.3f}\n'
                         f'MAE: {mae:.3f}\n'
                         f'Mean: {mean_residual:.3f}')
            
            ax6.text(
                0.05, 0.95, stats_text, 
                transform=ax6.transAxes, 
                fontsize=9, fontweight='medium', 
                va='top',
                bbox=dict(
                    boxstyle='round,pad=0.4', 
                    facecolor='white', 
                    alpha=0.95, 
                    edgecolor='#cccccc', 
                    linewidth=0.8
                ),
                color=colors['text']
            )
            
            ax6.set_xlabel('Fitted Values', fontweight='medium', color=colors['text'])
            ax6.set_ylabel('Residuals', fontweight='medium', color=colors['text'])
            ax6.set_title('Residuals Analysis\n(Goodness of Fit)', 
                         fontweight='bold', pad=20, color=colors['text'])
            
            # Refined legend
            legend6 = ax6.legend(
                frameon=True, 
                fancybox=False, 
                shadow=False, 
                loc='upper right', 
                fontsize=8, 
                framealpha=0.95,
                edgecolor='#dddddd',
                facecolor='white'
            )
            legend6.get_frame().set_linewidth(0.8)
        else:
            # Elegant unavailable message
            ax6.text(
                0.5, 0.5, 
                'Residuals Analysis\nUnavailable\n\nNo Power-Law Fit\nAvailable', 
                ha='center', va='center', 
                transform=ax6.transAxes,
                fontsize=11, fontweight='medium',
                bbox=dict(
                    boxstyle='round,pad=0.8', 
                    facecolor='#f5f5f5', 
                    alpha=0.9, 
                    edgecolor='#cccccc', 
                    linewidth=1
                ),
                color=colors['text']
            )
            ax6.set_title('Residuals Analysis', fontweight='bold', pad=20, color=colors['text'])
        
        ax6.grid(True, alpha=0.15, linestyle='-', linewidth=0.6, color='#cccccc')
        sns.despine(ax=ax6, top=True, right=True)        
        # ═══════════════════════════════════════════════════════════════════════════════
        # PANEL 7: Ultra-Modern Executive Summary Dashboard (Publication Quality)
        # ═══════════════════════════════════════════════════════════════════════════════
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        ax7.set_facecolor('white')
        
        # Import datetime for timestamp
        from datetime import datetime
        
        # Create sophisticated summary sections with modern design
        summary_sections = []
        
        # Section 1: Dataset Intelligence
        summary_sections.append({
            'title': '📊 DATASET INTELLIGENCE',
            'content': [
                f'🔢 Total Events: {alpha_stats["count"]:,}',
                f'📏 Alpha Range: [{alpha_stats["min"]:.3f}, {alpha_stats["max"]:.3f}]',
                f'📈 Mean α: {alpha_stats["mean"]:.3f} ± {alpha_stats["std"]:.3f}',
                f'📊 Median α: {alpha_stats.get("median", np.median(alpha_values)):.3f}',
                f'📦 IQR: [{np.percentile(alpha_values, 25):.3f}, {np.percentile(alpha_values, 75):.3f}]'
            ],
            'color_scheme': {'bg': '#e8f4fd', 'border': '#1976d2', 'text': '#0d47a1'}
        })
        
        # Section 2: Statistical Profile
        skewness = stats.skew(alpha_values)
        kurtosis = stats.kurtosis(alpha_values)
        cv = alpha_stats['std'] / alpha_stats['mean']  # Coefficient of variation
        
        summary_sections.append({
            'title': '📈 STATISTICAL PROFILE',
            'content': [
                f'📐 Skewness: {skewness:.3f}',
                f'   {"→ Right-skewed" if skewness > 0.5 else "→ Left-skewed" if skewness < -0.5 else "→ Symmetric"}',
                f'📊 Kurtosis: {kurtosis:.3f}',
                f'   {"→ Heavy-tailed" if kurtosis > 1 else "→ Light-tailed" if kurtosis < -1 else "→ Normal-tailed"}',
                f'� Coeff. Variation: {cv:.3f}'
            ],
            'color_scheme': {'bg': '#f3e5f5', 'border': '#7b1fa2', 'text': '#4a148c'}
        })
        
        # Section 3: Power-Law Intelligence
        if not np.isnan(measured_alpha):
            fit_quality_emoji = ("🟢" if power_law.get('r_squared', 0) > 0.8 
                               else "🟡" if power_law.get('r_squared', 0) > 0.6 
                               else "🔴")
            summary_sections.append({
                'title': '🔬 POWER-LAW INTELLIGENCE',
                'content': [
                    f'{fit_quality_emoji} Measured β: {measured_alpha:.3f} ± {alpha_err:.3f}',
                    f'📈 Correlation R²: {power_law.get("r_squared", 0):.3f}',
                    f'✅ Fit Assessment: {power_law.get("fit_quality", "unknown").title()}',
                    f'📊 Data Points: {power_law.get("n_points", 0)}',
                    f'📉 P-value: {power_law.get("p_value", np.nan):.1e}' if not np.isnan(power_law.get("p_value", np.nan)) else '📉 P-value: N/A'
                ],
                'color_scheme': {'bg': '#e8f5e8', 'border': '#388e3c', 'text': '#1b5e20'}
            })
        else:
            summary_sections.append({
                'title': '🔬 POWER-LAW INTELLIGENCE',
                'content': [
                    '🔴 Status: Analysis Failed',
                    f'⚠️ Issue: {power_law.get("fit_quality", "Unknown error")}',
                    f'📊 Available Points: {power_law.get("n_points", 0)}',
                    '💡 Recommendation: Collect more data',
                    '🎯 Min Required: ~10-15 points'
                ],
                'color_scheme': {'bg': '#fff3e0', 'border': '#f57c00', 'text': '#e65100'}
            })
        
        # Section 4: Parker's Hypothesis Verdict
        if nanoflare['is_nanoflare']:
            verdict_sections = {
                'title': '🎯 PARKER\'S HYPOTHESIS VERDICT',
                'content': [
                    '🎉 NANOFLARES DETECTED!',
                    '🔥 CORONAL HEATING CAPABLE',
                    f'🎯 Detection Confidence: {nanoflare["detection_confidence"]:.1%}',
                    f'📈 Statistical Significance: {nanoflare.get("z_score", 0):.2f}σ' if not np.isnan(nanoflare.get("z_score", np.nan)) else '📈 Significance: Assessment pending',
                    f'⬆️ Threshold Excess: +{measured_alpha - self.alpha_threshold:.3f}' if not np.isnan(measured_alpha) else '⬆️ Threshold: Under evaluation'
                ],
                'color_scheme': {'bg': '#e8f5e8', 'border': '#4caf50', 'text': '#2e7d32'}
            }
        else:
            verdict_sections = {
                'title': '🎯 PARKER\'S HYPOTHESIS VERDICT',
                'content': [
                    '❌ NO NANOFLARES DETECTED',
                    '❄️ INSUFFICIENT FOR CORONAL HEATING',
                    f'📋 Primary Reason: {nanoflare["reason"].replace("_", " ").title()}',
                    f'🎯 Detection Confidence: {nanoflare["detection_confidence"]:.1%}',
                    f'⬇️ Threshold Deficit: -{self.alpha_threshold - measured_alpha:.3f}' if not np.isnan(measured_alpha) else '⬇️ Threshold: Under evaluation'
                ],
                'color_scheme': {'bg': '#ffebee', 'border': '#f44336', 'text': '#c62828'}
            }
        summary_sections.append(verdict_sections)
        
        # Section 5: Analysis Metadata
        summary_sections.append({
            'title': '⚙️ ANALYSIS METADATA',
            'content': [
                f'🎯 Parker Threshold: α = {self.alpha_threshold:.1f} ± {self.alpha_uncertainty:.3f}',
                f'🔥 Heating Criterion: α > {self.alpha_threshold:.1f}',
                f'📊 Confidence Level: 95%',
                f'🕒 Analysis Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                '🧑‍🔬 Reference: Parker (1988) Theory'
            ],
            'color_scheme': {'bg': '#fafafa', 'border': '#9e9e9e', 'text': '#424242'}
        })
        
        # Ultra-modern section layout with sophisticated typography
        section_width = 0.19
        section_positions = [0.01, 0.205, 0.40, 0.595, 0.79]
        
        for i, (section, x_pos) in enumerate(zip(summary_sections, section_positions)):
            # Create section background with subtle styling
            section_bg = plt.Rectangle(
                (x_pos, 0.05), section_width, 0.9,
                transform=ax7.transAxes,
                facecolor=section['color_scheme']['bg'],
                edgecolor=section['color_scheme']['border'],
                linewidth=1.5,
                alpha=0.9,
                zorder=1
            )
            ax7.add_patch(section_bg)
            
            # Section title with modern typography
            ax7.text(
                x_pos + section_width/2, 0.88,
                section['title'],
                transform=ax7.transAxes,
                fontsize=10,
                fontweight='bold',
                ha='center',
                va='top',
                color=section['color_scheme']['text'],
                zorder=3
            )
            
            # Section content with refined spacing
            content_text = '\n'.join(section['content'])
            ax7.text(
                x_pos + 0.01, 0.78,
                content_text,
                transform=ax7.transAxes,
                fontsize=8,
                fontfamily='monospace',
                ha='left',
                va='top',
                color=section['color_scheme']['text'],
                zorder=3,
                linespacing=1.4
            )
        
        # Ultra-modern main title with sophisticated styling
        main_title = fig.suptitle(
            'Parker\'s Solar Nanoflare Hypothesis Analysis\n'
            'Testing Coronal Heating Capability via Power-Law Index (α > 2.0)', 
            fontsize=18, 
            fontweight='bold', 
            y=0.97,
            color=colors['text'],
            bbox=dict(
                boxstyle='round,pad=0.8', 
                facecolor='white', 
                alpha=0.95, 
                edgecolor='#cccccc', 
                linewidth=1.5
            )
        )
        
        # Sophisticated subtitle with methodology attribution
        fig.text(
            0.5, 0.935, 
            'Publication-Quality Analysis with Advanced Seaborn/Matplotlib Visualization', 
            ha='center', va='center', 
            fontsize=12, 
            style='italic', 
            color='#666666',
            alpha=0.9
        )
        
        # Refined layout adjustment
        plt.tight_layout(rect=[0, 0.02, 1, 0.91])
        
        # Save with publication-quality settings
        if output_path:
            plt.savefig(
                output_path, 
                dpi=300, 
                bbox_inches='tight', 
                facecolor='white',
                edgecolor='none', 
                pad_inches=0.3,
                metadata={
                    'Title': 'Parker Nanoflare Analysis',
                    'Author': 'Advanced Solar Physics Analysis',
                    'Subject': 'Coronal Heating via Nanoflare Detection',
                    'Creator': 'Python/Seaborn/Matplotlib'
                }
            )
            print(f"✅ Publication-quality plot saved to: {output_path}")
        
        plt.show()
        
        return fig
            
def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description='Solar Nanoflare Detection Analysis')
    parser.add_argument('--features_file', default='output/flare_features.csv',
                       help='Path to flare features CSV file')
    parser.add_argument('--alpha_column', default='alpha',
                       help='Column name for alpha values')
    parser.add_argument('--output_dir', default='output',
                       help='Output directory for results')    
    parser.add_argument('--alpha_threshold', type=float, default=2.0,
                       help='Alpha threshold for Parker\'s nanoflare detection (default: 2.0)')
    parser.add_argument('--alpha_uncertainty', type=float, default=0.03,
                       help='Alpha threshold uncertainty (default: 0.03)')
    
    args = parser.parse_args()
    
    # Load data
    try:
        print(f"Loading flare features from {args.features_file}")
        df = pd.read_csv(args.features_file)
        print(f"Loaded {len(df)} flare records")
        
        if args.alpha_column not in df.columns:
            print(f"Error: Column '{args.alpha_column}' not found in data")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Get alpha values
        alpha_values = df[args.alpha_column].dropna()
        print(f"Found {len(alpha_values)} valid alpha values")
        
        if len(alpha_values) < 10:
            print("Error: Insufficient alpha values for analysis")
            return
        
        # Initialize analyzer
        analyzer = NanoflareAnalyzer(
            alpha_threshold=args.alpha_threshold,
            alpha_uncertainty=args.alpha_uncertainty
        )
        
        # Perform analysis
        print("\nPerforming nanoflare detection analysis...")
        analysis_result = analyzer.analyze_alpha_predictions(alpha_values)
          # Print results
        print(f"\n{'='*70}")
        print("PARKER'S NANOFLARE HYPOTHESIS - CORONAL HEATING ANALYSIS")
        print(f"{'='*70}")
        
        alpha_stats = analysis_result['alpha_statistics']
        power_law = analysis_result['power_law_analysis']
        nanoflare = analysis_result['nanoflare_detection']
        
        print(f"Alpha Statistics:")
        print(f"  Mean: {alpha_stats['mean']:.4f} ± {alpha_stats['std']:.4f}")
        print(f"  Range: [{alpha_stats['min']:.4f}, {alpha_stats['max']:.4f}]")
        print(f"  Count: {alpha_stats['count']}")
        
        print(f"\nPower-Law Analysis:")
        if not np.isnan(power_law.get('alpha', np.nan)):
            print(f"  Measured β: {power_law['alpha']:.4f} ± {power_law.get('alpha_error', 0):.4f}")
            print(f"  R-squared: {power_law.get('r_squared', 0):.4f}")
            print(f"  Fit quality: {power_law.get('fit_quality', 'unknown')}")
        else:
            print(f"  Fit failed: {power_law.get('fit_quality', 'unknown')}")
        print(f"\nParker's Nanoflare Detection:")
        print(f"  Parker Threshold: {args.alpha_threshold:.1f} ± {args.alpha_uncertainty:.3f}")
        if nanoflare['is_nanoflare']:
            print(f"  🟢 NANOFLARES DETECTED - CORONAL HEATING CAPABLE!")
            print(f"  Detection confidence: {nanoflare['detection_confidence']:.1%}")
            if not np.isnan(nanoflare.get('z_score', np.nan)):
                print(f"  Statistical significance: {nanoflare['z_score']:.2f}σ")
            if not np.isnan(power_law.get('alpha', np.nan)):
                excess = power_law['alpha'] - args.alpha_threshold
                print(f"  Exceeds Parker threshold by: {excess:.3f}")
                print(f"  🔥 Nanoflares can power solar coronal heating!")
        else:
            print(f"  🔴 No nanoflares detected - Insufficient for coronal heating")
            print(f"  Reason: {nanoflare['reason'].replace('_', ' ')}")
            print(f"  Detection confidence: {nanoflare['detection_confidence']:.1%}")
            if not np.isnan(power_law.get('alpha', np.nan)):
                deficit = args.alpha_threshold - power_law['alpha'] 
                print(f"  Below Parker threshold by: {deficit:.3f}")
                print(f"  ❄️ Cannot power coronal heating via Parker's mechanism")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create visualization
        print(f"\nCreating analysis visualization...")
        plot_path = output_dir / 'nanoflare_analysis.png'
        analyzer.plot_analysis(analysis_result, alpha_values, plot_path)
        
        # Save detailed results
        results_path = output_dir / 'nanoflare_analysis_results.json'
          # Make results JSON serializable
        def make_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif pd.isna(obj):
                return None
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            else:
                return obj
        
        serializable_results = make_serializable(analysis_result)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Detailed results saved to {results_path}")
        print(f"\nParker's Nanoflare Hypothesis analysis complete!")
        print(f"Coronal heating capability assessed based on α > {args.alpha_threshold:.1f} criterion")
        
    except FileNotFoundError:
        print(f"Error: File {args.features_file} not found")
        print("Please run the feature extraction step first:")
        print("python src/feature_extract.py --fits_dir fits --output_dir output --target alpha")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

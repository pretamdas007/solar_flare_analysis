"""
Enhanced main analysis pipeline for solar flare studies
Integrates advanced ML models for nanoflare detection and energy analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processing.data_loader import GOESDataLoader
from src.ml_models.enhanced_flare_analysis import (
    EnhancedFlareDecompositionModel,
    NanoflareDetector, 
    FlareEnergyAnalyzer
)
from src.visualization.plotting import FlareVisualization


class EnhancedSolarFlareAnalyzer:
    """
    Enhanced solar flare analysis pipeline with advanced ML capabilities
    """
    
    def __init__(self, data_path=None, output_dir='output'):
        """
        Initialize the enhanced analyzer
        
        Parameters
        ----------
        data_path : str, optional
            Path to GOES data files
        output_dir : str, optional
            Directory for output files and plots
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.data_loader = GOESDataLoader()
        self.ml_model = None
        self.nanoflare_detector = NanoflareDetector()
        self.energy_analyzer = FlareEnergyAnalyzer()
        self.visualizer = FlareVisualization()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Analysis results storage
        self.results = {
            'raw_data': None,
            'preprocessed_data': None,
            'ml_predictions': None,
            'nanoflare_analysis': None,
            'energy_analysis': None,
            'statistical_analysis': None,
            'corona_heating_assessment': None
        }
    
    def load_and_preprocess_data(self, start_date=None, end_date=None, 
                                satellite='GOES-16', resample_freq='1min'):
        """
        Load and preprocess GOES XRS data
        
        Parameters
        ----------
        start_date : str or datetime, optional
            Start date for data loading
        end_date : str or datetime, optional
            End date for data loading
        satellite : str, optional
            GOES satellite identifier
        resample_freq : str, optional
            Resampling frequency for data
            
        Returns
        -------
        pandas.DataFrame
            Preprocessed data
        """
        print("Loading and preprocessing GOES XRS data...")
        
        # Load data
        if self.data_path:
            data = self.data_loader.load_from_files(
                self.data_path, start_date=start_date, end_date=end_date
            )
        else:
            data = self.data_loader.download_goes_data(
                start_date=start_date, end_date=end_date, satellite=satellite
            )
        
        if data is None or len(data) == 0:
            raise ValueError("No data loaded. Check data path or date range.")
        
        # Preprocess data
        processed_data = self.data_loader.preprocess_data(
            data, resample_freq=resample_freq
        )
        
        self.results['raw_data'] = data
        self.results['preprocessed_data'] = processed_data
        
        print(f"Loaded {len(processed_data)} data points from {processed_data.index[0]} to {processed_data.index[-1]}")
        
        return processed_data
    
    def initialize_ml_model(self, sequence_length=256, max_flares=5):
        """
        Initialize and build the enhanced ML model
        
        Parameters
        ----------
        sequence_length : int, optional
            Length of input sequences for the model
        max_flares : int, optional
            Maximum number of overlapping flares to detect
        """
        print("Initializing enhanced ML model...")
        
        # Determine number of features from data
        if self.results['preprocessed_data'] is not None:
            n_features = len([col for col in self.results['preprocessed_data'].columns 
                            if 'xrs' in col.lower()])
        else:
            n_features = 2  # Default for XRS-A and XRS-B
        
        self.ml_model = EnhancedFlareDecompositionModel(
            sequence_length=sequence_length,
            n_features=n_features,
            max_flares=max_flares
        )
        
        # Build the model
        self.ml_model.build_enhanced_model()
        
        print(f"Model initialized with {sequence_length} sequence length and {max_flares} max flares")
        print(f"Model summary:")
        self.ml_model.model.summary()
    
    def train_ml_model(self, use_synthetic_data=True, n_synthetic_samples=5000,
                      validation_split=0.2, epochs=100, batch_size=32):
        """
        Train the enhanced ML model
        
        Parameters
        ----------
        use_synthetic_data : bool, optional
            Whether to use synthetic data for training
        n_synthetic_samples : int, optional
            Number of synthetic samples to generate
        validation_split : float, optional
            Fraction of data for validation
        epochs : int, optional
            Number of training epochs
        batch_size : int, optional
            Training batch size
        """
        print("Training enhanced ML model...")
        
        if self.ml_model is None:
            raise ValueError("ML model not initialized. Call initialize_ml_model() first.")
        
        if use_synthetic_data:
            print(f"Generating {n_synthetic_samples} synthetic training samples...")
            X_train, y_train = self.ml_model.generate_enhanced_synthetic_data(
                n_samples=n_synthetic_samples
            )
        else:
            # Prepare real data for training (if available)
            X_train, y_train = self._prepare_real_training_data()
        
        # Train the model
        history = self.ml_model.train_enhanced_model(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Plot training history
        fig = self.ml_model.plot_enhanced_training_history()
        if fig:
            fig.savefig(os.path.join(self.output_dir, 'training_history.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        print("Model training completed!")
        
        return history
    
    def analyze_solar_flares(self, plot_results=True, save_results=True):
        """
        Perform comprehensive solar flare analysis
        
        Parameters
        ----------
        plot_results : bool, optional
            Whether to create visualization plots
        save_results : bool, optional
            Whether to save analysis results
            
        Returns
        -------
        dict
            Comprehensive analysis results
        """
        print("Performing comprehensive solar flare analysis...")
        
        if self.results['preprocessed_data'] is None:
            raise ValueError("No data available. Load data first.")
        
        if self.ml_model is None or self.ml_model.model is None:
            raise ValueError("ML model not trained. Train model first.")
        
        # Prepare data for ML analysis
        time_series_data = self._prepare_time_series_for_analysis()
        
        # ML-based flare decomposition
        print("Applying ML model for flare decomposition...")
        ml_predictions = self._apply_ml_decomposition(time_series_data)
        self.results['ml_predictions'] = ml_predictions
        
        # Nanoflare detection
        print("Detecting nanoflares...")
        nanoflare_results = self._detect_nanoflares(time_series_data)
        self.results['nanoflare_analysis'] = nanoflare_results
        
        # Energy analysis
        print("Analyzing flare energies and statistics...")
        energy_results = self._analyze_flare_energies(ml_predictions, nanoflare_results)
        self.results['energy_analysis'] = energy_results
        
        # Statistical analysis
        print("Performing statistical analysis...")
        statistical_results = self._perform_statistical_analysis()
        self.results['statistical_analysis'] = statistical_results
        
        # Corona heating assessment
        print("Assessing corona heating contribution...")
        heating_assessment = self._assess_corona_heating()
        self.results['corona_heating_assessment'] = heating_assessment
        
        # Generate visualizations
        if plot_results:
            self._create_comprehensive_visualizations()
        
        # Save results
        if save_results:
            self._save_analysis_results()
        
        print("Analysis completed!")
        
        return self.results
    
    def _prepare_time_series_for_analysis(self):
        """Prepare time series data for ML analysis"""
        data = self.results['preprocessed_data']
        
        # Extract XRS channels
        xrs_columns = [col for col in data.columns if 'xrs' in col.lower()]
        if len(xrs_columns) == 0:
            raise ValueError("No XRS data columns found")
        
        # Convert to numpy array
        time_series = data[xrs_columns].values
        
        # Handle missing values
        time_series = np.nan_to_num(time_series, nan=0.0)
        
        return time_series
    
    def _apply_ml_decomposition(self, time_series_data):
        """Apply ML model to decompose flares"""
        # Create sliding windows for analysis
        window_size = self.ml_model.sequence_length
        step_size = window_size // 4
        
        n_windows = max(1, (len(time_series_data) - window_size) // step_size + 1)
        
        # Prepare windows
        windows = []
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = min(start_idx + window_size, len(time_series_data))
            
            if end_idx - start_idx == window_size:
                window = time_series_data[start_idx:end_idx]
                windows.append(window)
        
        if len(windows) == 0:
            # Handle case where data is shorter than window size
            padded_data = np.zeros((window_size, time_series_data.shape[1]))
            padded_data[:len(time_series_data)] = time_series_data
            windows = [padded_data]
        
        windows = np.array(windows)
        
        # Make predictions
        predictions = self.ml_model.predict_enhanced(windows)
        
        return predictions
    
    def _detect_nanoflares(self, time_series_data):
        """Detect nanoflares in the time series"""
        # Use the primary channel (typically XRS-A)
        primary_channel = time_series_data[:, 0] if time_series_data.shape[1] > 0 else time_series_data.flatten()
        
        # Detect nanoflares
        nanoflare_results = self.nanoflare_detector.detect_nanoflares(
            primary_channel, sampling_rate=1/60  # Assuming 1-minute data
        )
        
        return nanoflare_results
    
    def _analyze_flare_energies(self, ml_predictions, nanoflare_results):
        """Analyze flare energies using both ML and traditional methods"""
        # Extract energies from ML predictions
        ml_energies = ml_predictions['energy_estimates'].flatten()
        ml_energies = ml_energies[ml_energies > 0]  # Remove zero/negative energies
        
        # Extract energies from nanoflare detection
        nano_energies = nanoflare_results.get('energies', [])
        
        # Combine all energies
        all_energies = np.concatenate([ml_energies, nano_energies]) if len(nano_energies) > 0 else ml_energies
        
        # Prepare flare data for analysis
        flare_data = {
            'energies': all_energies,
            'ml_energies': ml_energies,
            'nano_energies': nano_energies
        }
        
        # Perform comprehensive energy analysis
        energy_analysis = self.energy_analyzer.analyze_energy_distribution(flare_data)
        
        return energy_analysis
    
    def _perform_statistical_analysis(self):
        """Perform statistical analysis of flare properties"""
        ml_predictions = self.results['ml_predictions']
        nanoflare_results = self.results['nanoflare_analysis']
        
        # Analyze flare statistics from ML model
        ml_stats = self.ml_model.analyze_flare_statistics(
            ml_predictions, len(self.results['preprocessed_data'])
        )
        
        # Combine with nanoflare statistics
        statistical_results = {
            'ml_analysis': ml_stats,
            'nanoflare_detection': {
                'total_nanoflares': nanoflare_results.get('total_count', 0),
                'total_nano_energy': nanoflare_results.get('total_energy', 0),
                'alpha_values': nanoflare_results.get('alpha_values', []),
                'mean_alpha': np.mean(nanoflare_results.get('alpha_values', [0])),
                'alpha_threshold_exceeded': np.sum(np.abs(nanoflare_results.get('alpha_values', [0])) > 2)
            }
        }
        
        return statistical_results
    
    def _assess_corona_heating(self):
        """Assess corona heating contribution based on analysis results"""
        energy_analysis = self.results['energy_analysis']
        statistical_analysis = self.results['statistical_analysis']
        
        # Get power law analysis
        power_law_results = energy_analysis.get('power_law_analysis', {})
        nanoflare_analysis = energy_analysis.get('nanoflare_analysis', {})
        
        # Get corona heating assessment from energy analyzer
        heating_assessment = energy_analysis.get('corona_heating_assessment', {})
        
        # Enhance with additional criteria
        alpha = power_law_results.get('alpha')
        nanoflare_fraction = nanoflare_analysis.get('nanoflare_fraction', 0)
        alpha_threshold_count = statistical_analysis['nanoflare_detection']['alpha_threshold_exceeded']
        
        # Enhanced assessment
        enhanced_assessment = heating_assessment.copy()
        enhanced_assessment.update({
            'nanoflare_fraction': nanoflare_fraction,
            'alpha_threshold_events': alpha_threshold_count,
            'heating_contribution_score': self._calculate_heating_score(
                alpha, nanoflare_fraction, alpha_threshold_count
            )
        })
        
        return enhanced_assessment
    
    def _calculate_heating_score(self, alpha, nanoflare_fraction, alpha_threshold_count):
        """Calculate a score for nanoflare heating contribution"""
        score = 0.0
        
        # Alpha contribution (0-40 points)
        if alpha is not None:
            if alpha > 2.5:
                score += 40
            elif alpha > 2.0:
                score += 30
            elif alpha > 1.5:
                score += 20
            else:
                score += 10
        
        # Nanoflare fraction contribution (0-30 points)
        score += min(30, nanoflare_fraction * 30)
        
        # Alpha threshold events contribution (0-30 points)
        if alpha_threshold_count > 0:
            score += min(30, alpha_threshold_count * 5)
        
        return min(100, score)  # Cap at 100
    
    def _create_comprehensive_visualizations(self):
        """Create comprehensive visualization plots"""
        print("Creating visualizations...")
        
        # Time series plot with detected flares
        self._plot_time_series_with_flares()
        
        # Energy distribution plots
        self._plot_energy_distributions()
        
        # Statistical analysis plots
        self._plot_statistical_analysis()
        
        # Corona heating assessment
        self._plot_corona_heating_assessment()
        
        print("Visualizations saved to output directory")
    
    def _plot_time_series_with_flares(self):
        """Plot time series data with detected flares"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        data = self.results['preprocessed_data']
        nanoflare_results = self.results['nanoflare_analysis']
        
        # Get time array
        time_index = np.arange(len(data))
        
        # Plot XRS-A channel
        xrs_a_col = [col for col in data.columns if 'xrs_a' in col.lower() or 'xrsa' in col.lower()]
        if xrs_a_col:
            axes[0].plot(time_index, data[xrs_a_col[0]], 'b-', alpha=0.7, label='XRS-A')
            axes[0].set_ylabel('Flux (W/m²)')
            axes[0].set_title('GOES XRS-A Channel with Detected Flares')
            axes[0].set_yscale('log')
            axes[0].grid(True, alpha=0.3)
            
            # Mark nanoflare detections
            nano_peaks = nanoflare_results.get('peaks', [])
            if len(nano_peaks) > 0:
                axes[0].scatter(nano_peaks, data[xrs_a_col[0]].iloc[nano_peaks], 
                               color='red', s=50, alpha=0.8, label='Nanoflares')
            axes[0].legend()
        
        # Plot XRS-B channel
        xrs_b_col = [col for col in data.columns if 'xrs_b' in col.lower() or 'xrsb' in col.lower()]
        if xrs_b_col:
            axes[1].plot(time_index, data[xrs_b_col[0]], 'g-', alpha=0.7, label='XRS-B')
            axes[1].set_ylabel('Flux (W/m²)')
            axes[1].set_title('GOES XRS-B Channel')
            axes[1].set_yscale('log')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        
        # Plot alpha parameter evolution
        alpha_values = nanoflare_results.get('alpha_values', [])
        if len(alpha_values) > 0 and len(nano_peaks) == len(alpha_values):
            axes[2].scatter(nano_peaks, alpha_values, color='purple', alpha=0.7, s=30)
            axes[2].axhline(y=2, color='red', linestyle='--', alpha=0.8, label='α = 2 threshold')
            axes[2].axhline(y=-2, color='red', linestyle='--', alpha=0.8)
            axes[2].set_ylabel('Alpha Parameter')
            axes[2].set_xlabel('Time Index')
            axes[2].set_title('Alpha Parameter Evolution')
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'time_series_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_energy_distributions(self):
        """Plot energy distribution analysis"""
        energy_analysis = self.results['energy_analysis']
        
        # Get energies from analysis
        all_energies = []
        if 'ml_energies' in energy_analysis.get('basic_statistics', {}):
            all_energies = energy_analysis['basic_statistics']['ml_energies']
        elif self.results['ml_predictions'] is not None:
            all_energies = self.results['ml_predictions']['energy_estimates'].flatten()
            all_energies = all_energies[all_energies > 0]
        
        if len(all_energies) > 0:
            fig = self.energy_analyzer.plot_comprehensive_analysis(
                energy_analysis, all_energies
            )
            if fig:
                fig.savefig(os.path.join(self.output_dir, 'energy_analysis.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close(fig)
    
    def _plot_statistical_analysis(self):
        """Plot statistical analysis results"""
        statistical_results = self.results['statistical_analysis']
        nanoflare_results = self.results['nanoflare_analysis']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Alpha parameter distribution
        alpha_values = nanoflare_results.get('alpha_values', [])
        if len(alpha_values) > 0:
            axes[0, 0].hist(alpha_values, bins=20, alpha=0.7, color='purple')
            axes[0, 0].axvline(x=2, color='red', linestyle='--', label='α = 2 threshold')
            axes[0, 0].axvline(x=-2, color='red', linestyle='--')
            axes[0, 0].set_xlabel('Alpha Parameter')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Alpha Parameter Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Flare count statistics
        ml_stats = statistical_results.get('ml_analysis', {})
        nano_stats = statistical_results.get('nanoflare_detection', {})
        
        categories = ['ML Detected', 'Nanoflares', 'Alpha > 2']
        counts = [
            ml_stats.get('total_flares', 0),
            nano_stats.get('total_nanoflares', 0),
            nano_stats.get('alpha_threshold_exceeded', 0)
        ]
        
        axes[0, 1].bar(categories, counts, alpha=0.7, color=['blue', 'red', 'purple'])
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Flare Detection Statistics')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Energy statistics
        energy_stats = self.results['energy_analysis'].get('basic_statistics', {})
        if energy_stats:
            stats_names = ['Total Energy', 'Mean Energy', 'Median Energy', 'Max Energy']
            stats_values = [
                energy_stats.get('total_energy', 0),
                energy_stats.get('mean_energy', 0),
                energy_stats.get('median_energy', 0),
                energy_stats.get('max_energy', 0)
            ]
            
            axes[1, 0].bar(range(len(stats_names)), np.log10(np.array(stats_values) + 1e-20), 
                          alpha=0.7, color='green')
            axes[1, 0].set_xticks(range(len(stats_names)))
            axes[1, 0].set_xticklabels(stats_names, rotation=45)
            axes[1, 0].set_ylabel('Log10(Energy)')
            axes[1, 0].set_title('Energy Statistics')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Corona heating assessment summary
        heating_assessment = self.results['corona_heating_assessment']
        assessment_text = f"""Heating Mechanism: {heating_assessment.get('heating_mechanism', 'unknown')}
Confidence: {heating_assessment.get('confidence', 'unknown')}
Nanoflare Potential: {heating_assessment.get('nanoflare_heating_potential', False)}
Heating Score: {heating_assessment.get('heating_contribution_score', 0):.1f}/100"""
        
        axes[1, 1].text(0.1, 0.5, assessment_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Corona Heating Assessment')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'statistical_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_corona_heating_assessment(self):
        """Create detailed corona heating assessment visualization"""
        heating_assessment = self.results['corona_heating_assessment']
        energy_analysis = self.results['energy_analysis']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Power law fit visualization
        power_law_results = energy_analysis.get('power_law_analysis', {})
        alpha = power_law_results.get('alpha')
        r_squared = power_law_results.get('r_squared')
        
        # Heating mechanism pie chart
        mechanism = heating_assessment.get('heating_mechanism', 'unknown')
        colors = {
            'nanoflare_dominated': 'red',
            'mixed': 'orange', 
            'large_flare_dominated': 'blue',
            'unknown': 'gray'
        }
        
        axes[0, 0].pie([1], labels=[mechanism.replace('_', ' ').title()], 
                       colors=[colors.get(mechanism, 'gray')], autopct='')
        axes[0, 0].set_title('Dominant Heating Mechanism')
        
        # Confidence and scoring
        confidence = heating_assessment.get('confidence', 'low')
        score = heating_assessment.get('heating_contribution_score', 0)
        
        axes[0, 1].bar(['Confidence', 'Heating Score'], 
                       [{'low': 1, 'medium': 2, 'high': 3}.get(confidence, 1), score/33.33], 
                       color=['blue', 'red'], alpha=0.7)
        axes[0, 1].set_ylabel('Normalized Score')
        axes[0, 1].set_title('Assessment Confidence & Heating Score')
        axes[0, 1].set_ylim(0, 3)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Power law parameters
        if alpha is not None and r_squared is not None:
            params_text = f"""Power Law Analysis:
α = {alpha:.2f}
R² = {r_squared:.3f}
Fit Quality: {power_law_results.get('fit_quality', 'unknown')}

Nanoflare Indicators:
• α > 2: {'Yes' if alpha > 2 else 'No'}
• Good Fit: {'Yes' if r_squared > 0.6 else 'No'}"""
        else:
            params_text = "Power Law Analysis:\nInsufficient data for fit"
        
        axes[1, 0].text(0.1, 0.5, params_text, transform=axes[1, 0].transAxes,
                        fontsize=11, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].axis('off')
        axes[1, 0].set_title('Power Law Analysis')
        
        # Nanoflare statistics
        nanoflare_analysis = energy_analysis.get('nanoflare_analysis', {})
        nano_fraction = nanoflare_analysis.get('nanoflare_fraction', 0)
        nano_count = nanoflare_analysis.get('nanoflare_count', 0)
        
        nano_text = f"""Nanoflare Analysis:
Count: {nano_count}
Fraction: {nano_fraction:.1%}
Heating Potential: {'High' if heating_assessment.get('nanoflare_heating_potential', False) else 'Low'}

Energy Threshold: {nanoflare_analysis.get('energy_threshold', 0):.0e}"""
        
        axes[1, 1].text(0.1, 0.5, nano_text, transform=axes[1, 1].transAxes,
                        fontsize=11, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Nanoflare Statistics')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'corona_heating_assessment.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_analysis_results(self):
        """Save analysis results to files"""
        # Save summary statistics
        summary = self._create_summary_report()
        
        with open(os.path.join(self.output_dir, 'analysis_summary.txt'), 'w') as f:
            f.write(summary)
        
        # Save detailed results as CSV if possible
        try:
            # Flare detection results
            if self.results['nanoflare_analysis']:
                nano_results = self.results['nanoflare_analysis']
                if len(nano_results.get('peaks', [])) > 0:
                    flare_df = pd.DataFrame({
                        'peak_index': nano_results['peaks'],
                        'alpha_value': nano_results.get('alpha_values', []),
                        'energy': nano_results.get('energies', [])
                    })
                    flare_df.to_csv(os.path.join(self.output_dir, 'detected_flares.csv'), 
                                   index=False)
        except Exception as e:
            print(f"Warning: Could not save detailed results to CSV: {e}")
    
    def _create_summary_report(self):
        """Create a text summary of the analysis"""
        summary = f"""
ENHANCED SOLAR FLARE ANALYSIS REPORT
===================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA SUMMARY:
- Analysis period: {self.results['preprocessed_data'].index[0]} to {self.results['preprocessed_data'].index[-1]}
- Total data points: {len(self.results['preprocessed_data'])}
- Data channels: {', '.join(self.results['preprocessed_data'].columns)}

FLARE DETECTION RESULTS:
"""
        
        # ML Analysis
        if self.results['statistical_analysis']:
            ml_stats = self.results['statistical_analysis'].get('ml_analysis', {})
            summary += f"- ML detected flares: {ml_stats.get('total_flares', 0)}\n"
            summary += f"- Total ML energy: {ml_stats.get('total_energy', 0):.2e}\n"
        
        # Nanoflare Analysis
        if self.results['nanoflare_analysis']:
            nano_results = self.results['nanoflare_analysis']
            summary += f"- Detected nanoflares: {nano_results.get('total_count', 0)}\n"
            summary += f"- Nanoflare total energy: {nano_results.get('total_energy', 0):.2e}\n"
            summary += f"- Mean alpha parameter: {np.mean(nano_results.get('alpha_values', [0])):.2f}\n"
        
        # Energy Analysis
        if self.results['energy_analysis']:
            energy_stats = self.results['energy_analysis'].get('basic_statistics', {})
            power_law = self.results['energy_analysis'].get('power_law_analysis', {})
            
            summary += f"\nENERGY DISTRIBUTION ANALYSIS:\n"
            summary += f"- Total energy: {energy_stats.get('total_energy', 0):.2e}\n"
            summary += f"- Mean energy: {energy_stats.get('mean_energy', 0):.2e}\n"
            summary += f"- Energy range: {energy_stats.get('energy_range', 0):.2e}\n"
            
            if power_law.get('alpha') is not None:
                summary += f"- Power law index (α): {power_law['alpha']:.2f}\n"
                summary += f"- Power law R²: {power_law.get('r_squared', 0):.3f}\n"
                summary += f"- Fit quality: {power_law.get('fit_quality', 'unknown')}\n"
        
        # Corona Heating Assessment
        if self.results['corona_heating_assessment']:
            heating = self.results['corona_heating_assessment']
            summary += f"\nCORONA HEATING ASSESSMENT:\n"
            summary += f"- Heating mechanism: {heating.get('heating_mechanism', 'unknown')}\n"
            summary += f"- Assessment confidence: {heating.get('confidence', 'unknown')}\n"
            summary += f"- Nanoflare heating potential: {heating.get('nanoflare_heating_potential', False)}\n"
            summary += f"- Heating contribution score: {heating.get('heating_contribution_score', 0):.1f}/100\n"
            
            if heating.get('heating_mechanism') == 'nanoflare_dominated':
                summary += "\n*** SIGNIFICANT FINDING ***\n"
                summary += "Analysis suggests nanoflare-dominated heating mechanism.\n"
                summary += "This indicates that small-scale magnetic reconnection events\n"
                summary += "may be the primary contributor to coronal heating in this dataset.\n"
        
        summary += f"\nANALYSIS COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return summary
    
    def _prepare_real_training_data(self):
        """Prepare real data for training (placeholder for future implementation)"""
        # This would involve creating labeled training data from real observations
        # For now, return None to use synthetic data
        return None, None


def main():
    """
    Main execution function for enhanced solar flare analysis
    """
    print("Enhanced Solar Flare Analysis Pipeline")
    print("=====================================")
    
    # Initialize analyzer
    analyzer = EnhancedSolarFlareAnalyzer(output_dir='enhanced_output')
    
    try:
        # Example analysis workflow
        
        # 1. Load and preprocess data
        # For demonstration, you would specify your data path or date range
        # data = analyzer.load_and_preprocess_data(
        #     start_date='2023-01-01',
        #     end_date='2023-01-07',
        #     satellite='GOES-16'
        # )
        
        # 2. Initialize and train ML model
        analyzer.initialize_ml_model(sequence_length=256, max_flares=5)
        analyzer.train_ml_model(
            use_synthetic_data=True,
            n_synthetic_samples=3000,
            epochs=50,
            batch_size=32
        )
        
        # For demonstration with synthetic data analysis
        print("\nGenerating synthetic data for demonstration...")
        
        # Create synthetic time series for analysis
        synthetic_data = pd.DataFrame({
            'xrs_a': np.random.lognormal(-8, 1, 1000) * 1e-6,
            'xrs_b': np.random.lognormal(-9, 1, 1000) * 1e-6
        }, index=pd.date_range('2023-01-01', periods=1000, freq='1min'))
        
        # Add some synthetic flare events
        for i in range(5):
            peak_idx = np.random.randint(100, 900)
            flare_magnitude = np.random.uniform(10, 100)
            
            # Add flare profile
            for j in range(max(0, peak_idx-30), min(1000, peak_idx+60)):
                decay_factor = np.exp(-(j - peak_idx) / 20) if j >= peak_idx else np.exp((j - peak_idx) / 10)
                synthetic_data.iloc[j, 0] += flare_magnitude * decay_factor * 1e-6
                synthetic_data.iloc[j, 1] += flare_magnitude * 0.7 * decay_factor * 1e-6
        
        analyzer.results['preprocessed_data'] = synthetic_data
        
        # 3. Perform comprehensive analysis
        results = analyzer.analyze_solar_flares(plot_results=True, save_results=True)
        
        print("\nAnalysis completed successfully!")
        print(f"Results saved to: {analyzer.output_dir}")
        
        # Print summary
        summary = analyzer._create_summary_report()
        print("\nANALYSIS SUMMARY:")
        print(summary)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

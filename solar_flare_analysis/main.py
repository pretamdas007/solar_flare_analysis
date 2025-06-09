"""
Enhanced main script for comprehensive solar flare analysis.
Integrates advanced ML models for nanoflare detection, energy analysis, and corona heating assessment.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from datetime import datetime, timedelta
import sys
import warnings
import traceback
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.abspath('c:\\Users\\srabani\\Desktop\\goesflareenv\\solar_flare_analysis'))

# Import project modules
from config import settings
from src.data_processing.data_loader import GOESDataLoader, load_goes_data, preprocess_xrs_data, remove_background
from src.flare_detection.traditional_detection import (detect_flare_peaks, define_flare_bounds, 
                                                    detect_overlapping_flares)
from src.ml_models.flare_decomposition import FlareDecompositionModel, reconstruct_flares
from src.ml_models.bayesian_flare_analysis import BayesianFlareAnalyzer 
from src.ml_models.bayesian_flare_analysis import BayesianFlareEnergyEstimator
from src.analysis.power_law import calculate_flare_energy, fit_power_law, compare_flare_populations
from src.visualization.plotting import (plot_xrs_time_series, plot_detected_flares, 
                                      plot_flare_decomposition, plot_power_law_comparison, FlareVisualization)

# Try to import enhanced modules (may not exist yet)
try:
    from src.ml_models.enhanced_flare_analysis import (
        EnhancedFlareDecompositionModel,
        NanoflareDetector, 
        FlareEnergyAnalyzer
    )
    ENHANCED_MODELS_AVAILABLE = True
except ImportError:
    print("Enhanced models not available - falling back to basic models")
    ENHANCED_MODELS_AVAILABLE = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced Solar Flare Analysis')
    parser.add_argument('--data', type=str, help='Path to NetCDF data file or directory',
                       default=None)
    parser.add_argument('--channel', type=str, choices=['A', 'B', 'a', 'b'], 
                       default='B', help='XRS channel to analyze')
    parser.add_argument('--train', action='store_true', 
                       help='Train the ML model with synthetic data')
    parser.add_argument('--train-enhanced', action='store_true',
                       help='Train the enhanced ML model with advanced features')
    parser.add_argument('--train-bayesian', action='store_true',
                       help='Train the Bayesian ML model with Monte Carlo sampling')
    parser.add_argument('--model', type=str, 
                       help='Path to save/load model',
                       default=os.path.join(settings.MODEL_DIR, 'flare_decomposition_model'))
    parser.add_argument('--enhanced-model', type=str,
                       help='Path to save/load enhanced model',
                       default=os.path.join(settings.MODEL_DIR, 'enhanced_flare_model'))
    parser.add_argument('--output', type=str,
                       help='Output directory for results',
                       default=settings.OUTPUT_DIR)
    parser.add_argument('--start-date', type=str,
                       help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date for analysis (YYYY-MM-DD)')
    parser.add_argument('--nanoflare-analysis', action='store_true',
                       help='Perform nanoflare detection and analysis')
    parser.add_argument('--corona-heating', action='store_true',
                       help='Assess corona heating contribution')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive analysis with all features')
    parser.add_argument('--generate-synthetic', action='store_true',
                       help='Generate synthetic data and save to CSV files')
    parser.add_argument('--synthetic-samples', type=int, default=5000,
                       help='Number of synthetic samples to generate')
    return parser.parse_args()


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
        self.data_loader = GOESDataLoader() if hasattr(GOESDataLoader, '__init__') else None
        self.ml_model = None
        
        # Initialize enhanced components if available
        if ENHANCED_MODELS_AVAILABLE:
            self.nanoflare_detector = NanoflareDetector()
            self.energy_analyzer = FlareEnergyAnalyzer()
            self.visualizer = FlareVisualization()
        else:
            self.nanoflare_detector = None
            self.energy_analyzer = None
            self.visualizer = None
        
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
        
        # Load data using enhanced loader if available
        if self.data_loader:
            if self.data_path:
                data = self.data_loader.load_from_files(
                    self.data_path, start_date=start_date, end_date=end_date
                )
            else:
                # Try to download data if loader supports it
                if hasattr(self.data_loader, 'download_goes_data'):
                    data = self.data_loader.download_goes_data(
                        start_date=start_date, end_date=end_date, satellite=satellite
                    )
                else:
                    data = None
        else:
            # Fall back to basic loader
            if self.data_path:
                data = load_goes_data(self.data_path)
            else:
                data = None
        
        if data is None or len(data) == 0:
            print("No data loaded. Check data path or date range.")
            return None
        
        # Preprocess data
        if self.data_loader and hasattr(self.data_loader, 'preprocess_data'):
            processed_data = self.data_loader.preprocess_data(
                data, resample_freq=resample_freq
            )
        else:
            # Fall back to basic preprocessing
            processed_data = preprocess_xrs_data(
                data, resample_freq=resample_freq, apply_quality_filter=True
            )
        
        self.results['raw_data'] = data
        self.results['preprocessed_data'] = processed_data
        
        print(f"Loaded {len(processed_data)} data points from {processed_data.index[0]} to {processed_data.index[-1]}")
        
        return processed_data
    
    def initialize_ml_model(self, sequence_length=256, max_flares=5, enhanced=False):
        """
        Initialize and build the ML model
        
        Parameters
        ----------
        sequence_length : int, optional
            Length of input sequences for the model
        max_flares : int, optional
            Maximum number of overlapping flares to detect
        enhanced : bool, optional
            Whether to use enhanced model if available
        """
        print("Initializing ML model...")
        
        # Determine number of features from data
        if self.results['preprocessed_data'] is not None:
            n_features = len([col for col in self.results['preprocessed_data'].columns 
                            if 'xrs' in col.lower()])
        else:
            n_features = 2  # Default for XRS-A and XRS-B
        
        # Use enhanced model if available and requested
        if enhanced and ENHANCED_MODELS_AVAILABLE:
            self.ml_model = EnhancedFlareDecompositionModel(
                sequence_length=sequence_length,
                n_features=n_features,
                max_flares=max_flares
            )
            self.ml_model.build_enhanced_model()
            print("Enhanced ML model initialized")
        else:
            # Fall back to basic model
            self.ml_model = FlareDecompositionModel(
                sequence_length=sequence_length,
                n_features=n_features,
                max_flares=max_flares
            )
            self.ml_model.build_model()
            print("Basic ML model initialized")
        
        print(f"Model initialized with {sequence_length} sequence length and {max_flares} max flares")
        if hasattr(self.ml_model, 'model') and self.ml_model.model:
            print("Model summary:")
            self.ml_model.model.summary()
    
    def train_ml_model(self, use_synthetic_data=True, n_synthetic_samples=5000,
                      validation_split=0.2, epochs=1, batch_size=32, enhanced=False):
        """
        Train the ML model
        
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
        enhanced : bool, optional
            Whether to use enhanced training methods
        """
        print("Training ML model...")
        
        if self.ml_model is None:
            raise ValueError("ML model not initialized. Call initialize_ml_model() first.")
        
        if use_synthetic_data:
            print(f"Generating {n_synthetic_samples} synthetic training samples...")
            if enhanced and hasattr(self.ml_model, 'generate_enhanced_synthetic_data'):
                X_train, y_train = self.ml_model.generate_enhanced_synthetic_data(
                    n_samples=n_synthetic_samples
                )
            else:
                X_train, y_train = self.ml_model.generate_synthetic_data(
                    n_samples=n_synthetic_samples
                )
        else:
            # Prepare real data for training (if available)
            X_train, y_train = self._prepare_real_training_data()
        
        # Train the model
        if enhanced and hasattr(self.ml_model, 'train_enhanced_model'):
            history = self.ml_model.train_enhanced_model(
                X_train, y_train,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size
            )
        else:
            # Use basic training
            X_val, y_val = self.ml_model.generate_synthetic_data(
                n_samples=n_synthetic_samples//5
            ) if use_synthetic_data else (None, None)
            
            history = self.ml_model.train(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=epochs,
                batch_size=batch_size
            )
        
        # Plot training history
        if hasattr(self.ml_model, 'plot_enhanced_training_history'):
            fig = self.ml_model.plot_enhanced_training_history()
        elif hasattr(self.ml_model, 'plot_training_history'):
            fig = self.ml_model.plot_training_history()
        else:
            fig = None
            
        if fig:
            fig.savefig(os.path.join(self.output_dir, 'training_history.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        print("Model training completed!")
        return history
    
    def save_synthetic_data_to_csv(self, n_samples=5000, output_dir=None, enhanced=False):
        """
        Generate and save synthetic data to CSV files in the data folder
        
        Parameters
        ----------
        n_samples : int, optional
            Number of synthetic samples to generate
        output_dir : str, optional
            Output directory for CSV files (defaults to data folder)
        enhanced : bool, optional
            Whether to use enhanced synthetic data generation
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), 'data')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating {n_samples} synthetic samples...")
        
        if self.ml_model is None:
            # Initialize a basic model for synthetic data generation
            self.initialize_ml_model()
        
        # Generate synthetic data
        if enhanced and hasattr(self.ml_model, 'generate_enhanced_synthetic_data'):
            X_data, y_data = self.ml_model.generate_enhanced_synthetic_data(
                n_samples=n_samples
            )
            prefix = "enhanced_synthetic"
        else:
            X_data, y_data = self.ml_model.generate_synthetic_data(
                n_samples=n_samples
            )
            prefix = "synthetic"
        
        # Save input data (X_data) - time series data
        print("Saving input time series data...")
        if len(X_data.shape) == 3:  # (samples, timesteps, features)
            # Reshape to 2D for CSV saving: (samples * timesteps, features + sample_id + timestep)
            n_samples_actual, timesteps, n_features = X_data.shape
            
            # Create a DataFrame with all time series data
            data_rows = []
            for sample_idx in range(n_samples_actual):
                for timestep in range(timesteps):
                    row = {
                        'sample_id': sample_idx,
                        'timestep': timestep,
                        'time_seconds': timestep * 60,  # Assuming 1-minute intervals
                    }
                    # Add feature columns (XRS channels)
                    for feature_idx in range(n_features):
                        if feature_idx == 0:
                            row['xrs_a_flux'] = X_data[sample_idx, timestep, feature_idx]
                        elif feature_idx == 1:
                            row['xrs_b_flux'] = X_data[sample_idx, timestep, feature_idx]
                        else:
                            row[f'feature_{feature_idx}'] = X_data[sample_idx, timestep, feature_idx]
                    data_rows.append(row)
            
            # Create DataFrame and save
            input_df = pd.DataFrame(data_rows)
            input_csv_path = os.path.join(output_dir, f'{prefix}_input_timeseries.csv')
            input_df.to_csv(input_csv_path, index=False)
            print(f"Input time series data saved to: {input_csv_path}")
            print(f"Shape: {input_df.shape}, Columns: {list(input_df.columns)}")
        
        # Save target data (y_data) - flare parameters
        print("Saving target flare parameters...")
        if len(y_data.shape) >= 2:  # Multiple outputs
            # Create target DataFrame
            target_rows = []
            for sample_idx in range(len(y_data)):
                if len(y_data.shape) == 3:  # (samples, max_flares, parameters)
                    max_flares, n_params = y_data.shape[1], y_data.shape[2]
                    for flare_idx in range(max_flares):
                        row = {
                            'sample_id': sample_idx,
                            'flare_id': flare_idx,
                        }
                        # Add flare parameters
                        for param_idx in range(n_params):
                            if param_idx == 0:
                                row['start_time'] = y_data[sample_idx, flare_idx, param_idx]
                            elif param_idx == 1:
                                row['end_time'] = y_data[sample_idx, flare_idx, param_idx]
                            elif param_idx == 2:
                                row['peak_time'] = y_data[sample_idx, flare_idx, param_idx]
                            elif param_idx == 3:
                                row['peak_flux'] = y_data[sample_idx, flare_idx, param_idx]
                            elif param_idx == 4:
                                row['energy'] = y_data[sample_idx, flare_idx, param_idx]
                            else:
                                row[f'param_{param_idx}'] = y_data[sample_idx, flare_idx, param_idx]
                        target_rows.append(row)
                else:  # (samples, parameters)
                    row = {'sample_id': sample_idx}
                    for param_idx in range(y_data.shape[1]):
                        if param_idx == 0:
                            row['start_time'] = y_data[sample_idx, param_idx]
                        elif param_idx == 1:
                            row['end_time'] = y_data[sample_idx, param_idx]
                        elif param_idx == 2:
                            row['peak_time'] = y_data[sample_idx, param_idx]
                        elif param_idx == 3:
                            row['peak_flux'] = y_data[sample_idx, param_idx]
                        elif param_idx == 4:
                            row['energy'] = y_data[sample_idx, param_idx]
                        else:
                            row[f'param_{param_idx}'] = y_data[sample_idx, param_idx]
                    target_rows.append(row)
            
            # Create DataFrame and save
            target_df = pd.DataFrame(target_rows)
            target_csv_path = os.path.join(output_dir, f'{prefix}_target_parameters.csv')
            target_df.to_csv(target_csv_path, index=False)
            print(f"Target parameters saved to: {target_csv_path}")
            print(f"Shape: {target_df.shape}, Columns: {list(target_df.columns)}")
        
        # Save summary statistics
        print("Saving summary statistics...")
        summary_data = {
            'generation_timestamp': [datetime.now().isoformat()],
            'n_samples': [n_samples if len(X_data.shape) == 3 else len(X_data)],
            'n_timesteps': [X_data.shape[1] if len(X_data.shape) == 3 else 1],
            'n_features': [X_data.shape[-1]],
            'enhanced_generation': [enhanced],
            'input_shape': [str(X_data.shape)],
            'target_shape': [str(y_data.shape)],
            'input_mean': [np.mean(X_data)],
            'input_std': [np.std(X_data)],
            'input_min': [np.min(X_data)],
            'input_max': [np.max(X_data)],
            'target_mean': [np.mean(y_data)],
            'target_std': [np.std(y_data)],
            'target_min': [np.min(y_data)],
            'target_max': [np.max(y_data)]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(output_dir, f'{prefix}_summary.csv')
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Summary statistics saved to: {summary_csv_path}")
        
        # Create a simple time series example for visualization
        print("Creating example time series...")
        if len(X_data.shape) == 3:
            # Take first sample for example
            example_sample = X_data[0]  # Shape: (timesteps, features)
            example_rows = []
            
            for timestep in range(example_sample.shape[0]):
                row = {
                    'time_minutes': timestep,
                    'time_seconds': timestep * 60,
                    'datetime': (datetime.now() + timedelta(minutes=timestep)).isoformat(),
                }
                # Add flux columns
                for feature_idx in range(example_sample.shape[1]):
                    if feature_idx == 0:
                        row['xrs_a_flux'] = example_sample[timestep, feature_idx]
                    elif feature_idx == 1:
                        row['xrs_b_flux'] = example_sample[timestep, feature_idx]
                    else:
                        row[f'feature_{feature_idx}'] = example_sample[timestep, feature_idx]
                example_rows.append(row)
            
            example_df = pd.DataFrame(example_rows)
            example_csv_path = os.path.join(output_dir, f'{prefix}_example_timeseries.csv')
            example_df.to_csv(example_csv_path, index=False)
            print(f"Example time series saved to: {example_csv_path}")
        
        print(f"\n✅ Synthetic data generation complete!")
        print(f"📁 Files saved to: {output_dir}")
        return {
            'input_data': X_data,
            'target_data': y_data,
            'output_directory': output_dir,
            'files_created': [
                f'{prefix}_input_timeseries.csv',
                f'{prefix}_target_parameters.csv',
                f'{prefix}_summary.csv',
                f'{prefix}_example_timeseries.csv'
            ]
        }
    def _prepare_real_training_data(self):
        """Prepare real GOES data for training"""
        if self.results['preprocessed_data'] is None:
            print("No preprocessed data available for training")
            return None, None
    
        data = self.results['preprocessed_data']
        print(f"Preparing training data from {len(data)} samples...")
    
        # Create sequences from real data
        sequence_length = self.ml_model.sequence_length
        X_sequences = []
        y_labels = []
    
        # Extract flux columns
        flux_cols = [col for col in data.columns if 'xrs' in col.lower()]
        if not flux_cols:
            print("No XRS flux columns found in data")
            return None, None
        
        print(f"Using flux columns: {flux_cols}")
        
        # Detect flares in the real data for labeling
        print("Detecting flares for labeling...")
        detected_flares = self._detect_flares_for_labeling(data, flux_cols)
        
        # Create sliding windows
        print(f"Creating sequences with length {sequence_length}...")
        for i in range(len(data) - sequence_length):
            sequence = data[flux_cols].iloc[i:i+sequence_length].values
            X_sequences.append(sequence)
        
            # Create labels based on detected flares in this sequence
            sequence_data = data.iloc[i:i+sequence_length]
            label = self._create_label_for_sequence(sequence_data, detected_flares)
            y_labels.append(label)
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(data) - sequence_length} sequences")
    
        print(f"Created {len(X_sequences)} training sequences")
        return np.array(X_sequences), np.array(y_labels)
    
    def analyze_solar_flares(self, plot_results=True, save_results=True, 
                           nanoflare_analysis=False, corona_heating=False):
        """
        Perform comprehensive solar flare analysis
        
        Parameters
        ----------
        plot_results : bool, optional
            Whether to create visualization plots
        save_results : bool, optional
            Whether to save analysis results
        nanoflare_analysis : bool, optional
            Whether to perform nanoflare analysis
        corona_heating : bool, optional
            Whether to assess corona heating
            
        Returns
        -------
        dict
            Analysis results
        """
        print("Running comprehensive solar flare analysis...")
        
        results = {
            'ml_predictions': {'energy_estimates': []},
            'nanoflare_analysis': {'total_count': 0},
            'corona_heating_assessment': {'heating_contribution_score': 0.0}
        }
        
        if self.results['preprocessed_data'] is None:
            print("No preprocessed data available for analysis")
            return results
        
        # Placeholder analysis - implement actual logic as needed
        print("Analysis completed (placeholder implementation)")
        
        return results
    
    def _detect_flares_for_labeling(self, data, flux_cols):
        """
        Detect flares in real data for creating training labels
        
        Parameters
        ----------
        data : pd.DataFrame
            Preprocessed GOES data
        flux_cols : list
            List of flux column names
            
        Returns
        -------
        list
            List of detected flare events with properties
        """
        from src.flare_detection.traditional_detection import detect_flare_peaks, define_flare_bounds
        
        all_flares = []
        
        for flux_col in flux_cols:
            try:
                # Detect peaks
                peaks = detect_flare_peaks(
                    data, 
                    flux_column=flux_col,
                    threshold_factor=3.0,
                    window_size=25
                )
                
                if len(peaks) > 0:
                    # Define flare bounds
                    flares = define_flare_bounds(
                        data,
                        flux_column=flux_col,
                        peak_indices=peaks['peak_index'].values,
                        start_threshold=0.5,
                        end_threshold=0.5,
                        min_duration='2min',
                        max_duration='6H'
                    )
                    
                    # Add channel information
                    flares['channel'] = flux_col
                    all_flares.append(flares)
                    
                    print(f"  Detected {len(flares)} flares in {flux_col}")
                    
            except Exception as e:
                print(f"  Failed to detect flares in {flux_col}: {e}")
                continue
        
        if all_flares:
            combined_flares = pd.concat(all_flares, ignore_index=True)
            print(f"Total detected flares: {len(combined_flares)}")
            return combined_flares
        else:
            print("No flares detected for labeling")
            return pd.DataFrame()
    
    def _create_label_for_sequence(self, sequence_data, detected_flares):
        """
        Create training labels for a data sequence based on detected flares
        
        Parameters
        ----------
        sequence_data : pd.DataFrame
            Time series data for this sequence
        detected_flares : pd.DataFrame
            DataFrame of detected flare events
            
        Returns
        -------
        np.array
            Label array for this sequence (flare parameters)
        """
        # Initialize label array - for basic model: [start_time, end_time, peak_time, peak_flux, energy]
        max_flares = getattr(self.ml_model, 'max_flares', 5)
        n_params = 5  # start, end, peak time (normalized), peak flux, energy
        label = np.zeros((max_flares, n_params))
        
        if len(detected_flares) == 0:
            return label
        
        # Get sequence time range
        seq_start_time = sequence_data.index[0]
        seq_end_time = sequence_data.index[-1]
        sequence_duration = (seq_end_time - seq_start_time).total_seconds()
        
        # Find flares that overlap with this sequence
        overlapping_flares = []
        
        for idx, flare in detected_flares.iterrows():
            flare_start = flare['start_time']
            flare_end = flare['end_time']
            
            # Check if flare overlaps with sequence
            if (flare_start <= seq_end_time and flare_end >= seq_start_time):
                overlapping_flares.append(flare)
        
        # Fill label array with overlapping flares (up to max_flares)
        for i, flare in enumerate(overlapping_flares[:max_flares]):
            try:
                # Normalize times to sequence duration (0-1)
                start_norm = max(0, (flare['start_time'] - seq_start_time).total_seconds() / sequence_duration)
                end_norm = min(1, (flare['end_time'] - seq_start_time).total_seconds() / sequence_duration)
                peak_norm = (flare['peak_time'] - seq_start_time).total_seconds() / sequence_duration
                
                # Clip to valid range
                start_norm = np.clip(start_norm, 0, 1)
                end_norm = np.clip(end_norm, 0, 1)
                peak_norm = np.clip(peak_norm, 0, 1)
                
                # Get peak flux (log scale for better numerical stability)
                peak_flux = flare['peak_flux']
                log_peak_flux = np.log10(max(peak_flux, 1e-10))  # Avoid log(0)
                
                # Estimate energy (simple approximation)
                duration_seconds = (flare['end_time'] - flare['start_time']).total_seconds()
                energy_estimate = peak_flux * duration_seconds
                log_energy = np.log10(max(energy_estimate, 1e-15))
                
                # Fill label array
                label[i] = [start_norm, end_norm, peak_norm, log_peak_flux, log_energy]
                
            except Exception as e:
                print(f"Warning: Failed to process flare {i}: {e}")
                continue
        
        return label.flatten()  # Flatten for model compatibility
def generate_synthetic_data(args):
    """Generate synthetic data and save to CSV files."""
    print("\n=== Generating Synthetic Data ===")
    
    # Initialize analyzer
    analyzer = EnhancedSolarFlareAnalyzer(
        data_path=args.data,
        output_dir=args.output
    )
    
    # Generate and save synthetic data
    try:
        result = analyzer.save_synthetic_data_to_csv(
            n_samples=args.synthetic_samples,
            output_dir=os.path.join(os.path.dirname(__file__), 'data'),
            enhanced=args.train_enhanced
        )
        
        print(f"\n✅ Successfully generated synthetic data!")
        print(f"📊 Generated {args.synthetic_samples} samples")
        print(f"📁 Saved to: {result['output_directory']}")
        print(f"📄 Files created:")
        for file_name in result['files_created']:
            print(f"   - {file_name}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error generating synthetic data: {e}")
        traceback.print_exc()
        return None


def train_enhanced_model(args):
    """Train the enhanced flare decomposition model."""
    print("\n=== Training Enhanced Flare Decomposition Model ===")
    
    # Initialize enhanced analyzer
    analyzer = EnhancedSolarFlareAnalyzer(
        data_path=args.data,
        output_dir=args.output
    )
    
    # Initialize ML model
    analyzer.initialize_ml_model(
        sequence_length=settings.ML_PARAMS.get('sequence_length', 256),
        max_flares=settings.ML_PARAMS.get('max_flares', 5),
        enhanced=True
    )
    
    # Train the model
    history = analyzer.train_ml_model(
        use_synthetic_data=True,
        n_synthetic_samples=5000,
        epochs=settings.ML_PARAMS.get('epochs', 100),
        batch_size=settings.ML_PARAMS.get('batch_size', 32),
        enhanced=True
    )
    
    # Save the model
    if hasattr(analyzer.ml_model, 'save_model'):
        model_path = args.enhanced_model + '.h5'
        analyzer.ml_model.save_model(model_path)
        print(f"Enhanced model saved to {model_path}")
    
    return analyzer


def run_comprehensive_analysis(args):
    """Run comprehensive solar flare analysis."""
    print("\n=== Comprehensive Solar Flare Analysis ===")
    
    # Initialize enhanced analyzer
    analyzer = EnhancedSolarFlareAnalyzer(
        data_path=args.data,
        output_dir=args.output
    )
    
    # Load and preprocess data
    print("Loading data...")
    try:
        data = analyzer.load_and_preprocess_data(
            start_date=args.start_date,
            end_date=args.end_date
        )
        if data is None:
            print("Failed to load data. Please check data path and format.")
            return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Initialize and train/load ML model
    print("Initializing ML model...")
    try:
        analyzer.initialize_ml_model(enhanced=args.train_enhanced or args.comprehensive)
        
        # Try to load existing model first
        model_loaded = False
        if not args.train and not args.train_enhanced:
            try:
                model_path = args.enhanced_model + '.h5' if args.train_enhanced or args.comprehensive else args.model + '.h5'
                if os.path.exists(model_path) and hasattr(analyzer.ml_model, 'load_model'):
                    analyzer.ml_model.load_model(model_path)
                    model_loaded = True
                    print(f"Loaded existing model from {model_path}")
            except Exception as e:
                print(f"Could not load existing model: {e}")
        
        # Train new model if needed
        if not model_loaded:
            print("Training new model...")
            analyzer.train_ml_model(enhanced=args.train_enhanced or args.comprehensive)
            
    except Exception as e:
        print(f"Error with ML model: {e}")
        return
    
    # Run comprehensive analysis
    print("Running analysis...")
    try:
        results = analyzer.analyze_solar_flares(
            plot_results=True,
            save_results=True,
            nanoflare_analysis=args.nanoflare_analysis or args.comprehensive,
            corona_heating=args.corona_heating or args.comprehensive
        )
        
        # Print summary
        print("\n=== Analysis Summary ===")
        if results.get('ml_predictions'):
            energy_estimates = results['ml_predictions'].get('energy_estimates', [])
            print(f"ML Energy Estimates: {len(energy_estimates)} predictions")
            if len(energy_estimates) > 0:
                print(f"Mean Energy: {np.mean(energy_estimates):.2e}")
        
        if results.get('nanoflare_analysis'):
            nano_count = results['nanoflare_analysis'].get('total_count', 0)
            print(f"Nanoflares Detected: {nano_count}")
        
        if results.get('corona_heating_assessment'):
            heating_score = results['corona_heating_assessment'].get('heating_contribution_score', 0)
            print(f"Corona Heating Score: {heating_score:.3f}")
        
        print(f"\nResults saved to: {args.output}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


def train_model(args):
    """Train the basic flare decomposition model with synthetic data."""
    print("\n=== Training Basic Flare Decomposition Model ===")
    
    # Create model
    model = FlareDecompositionModel(
        sequence_length=settings.ML_PARAMS['sequence_length'],
        n_features=settings.ML_PARAMS['n_features'],
        max_flares=settings.ML_PARAMS['max_flares'],
        dropout_rate=settings.ML_PARAMS.get('dropout_rate', 0.2)
    )
    
    # Build model architecture
    model.build_model()
    if hasattr(model, 'model') and model.model:
        model.model.summary()
    
    # Generate synthetic training data
    print("Generating synthetic training data...")
    X_train, y_train = model.generate_synthetic_data(n_samples=5000, noise_level=0.05)
    X_val, y_val = model.generate_synthetic_data(n_samples=1000, noise_level=0.05)
    
    # Train model
    print("Training model...")
    history = model.train(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=settings.ML_PARAMS['epochs'],
        batch_size=settings.ML_PARAMS['batch_size'],
        save_path=args.model
    )
    
    # Plot training history
    if hasattr(model, 'plot_training_history'):
        history_fig = model.plot_training_history()
        history_fig.savefig(os.path.join(args.output, 'basic_training_history.png'), 
                           dpi=settings.VISUALIZATION_PARAMS['dpi'])
        plt.close(history_fig)
    
    # Evaluate on test data
    print("Evaluating model...")
    X_test, y_test = model.generate_synthetic_data(n_samples=500, noise_level=0.08)
    if hasattr(model, 'evaluate'):
        eval_results = model.evaluate(X_test, y_test)
        print(f"Evaluation results: {eval_results}")
    
    # Save model
    if hasattr(model, 'save_model'):
        model.save_model(args.model + '.h5')
        print(f"Model saved to {args.model + '.h5'}")
    
    return model


def analyze_flares(args):
    """Analyze solar flares in GOES XRS data."""
    # Step 1: Load and preprocess data
    print("\n=== Loading and Preprocessing Data ===")
    if args.data:
        data_path = args.data
    else:
        # Use sample data if available, or exit if not
        sample_files = [f for f in os.listdir(settings.DATA_DIR) if f.endswith('.csv')]
        if sample_files:
            data_path = os.path.join(settings.DATA_DIR, sample_files[0])
            print(f"Using sample CSV data file: {data_path}")
        else:
            print("No CSV data file provided and no sample CSV files found.")
            print(f"Please place GOES XRS .csv files in {settings.DATA_DIR} or specify --data")
            return
      # Load data using CSV loader
    print(f"Loading data from: {data_path}")
    data = load_goes_data(data_path)
    if data is None:
        print("Failed to load data. Exiting.")
        return    # Preprocess data
    print(f"Preprocessing {args.channel} channel data...")
    df = preprocess_xrs_data(data, resample_freq='1min', apply_quality_filter=True, 
                            normalize=False, remove_background=False)
    
    if df is None:
        print("Failed to preprocess data. The data file may be empty or invalid.")
        print("Please check the CSV file at: solar_flare_analysis/data/xrsb2_flux_observed.csv")
        return
    
    print(f"Available columns: {df.columns.tolist()}")
    
    # Map channel name to column name
    channel_mapping = {
        'a': 'xrs_a',
        'b': 'xrs_b'
    }
    column_name = channel_mapping.get(args.channel.lower(), f'xrs_{args.channel.lower()}')
    
    # Add extensive debugging for data/column issues
    print(f"\n=== Data Loading Debug Information ===")
    print(f"Requested channel: {args.channel}")
    print(f"Channel mapping: {channel_mapping}")
    print(f"Mapped column name: {column_name}")
    print(f"Available columns: {df.columns.tolist()}")
    print(f"Data shape: {df.shape}")
    print(f"Data types:\n{df.dtypes}")
    print(f"First few rows:\n{df.head()}")
    
    # Check if the column exists
    if column_name not in df.columns:
        print(f"ERROR: Column '{column_name}' not found in data!")
        print("Available columns:", df.columns.tolist())
        
        # Try to find similar columns
        possible_cols = [col for col in df.columns if 'xrs' in col.lower() or 'flux' in col.lower()]
        if possible_cols:
            print(f"Possible flux columns found: {possible_cols}")
            column_name = possible_cols[0]  # Use the first available flux column
            print(f"Using column: {column_name}")
        else:
            print("No flux columns found! Please check your data file format.")
            return
    
    # Check data quality
    print(f"\n=== Data Quality Check ===")
    print(f"Column '{column_name}' statistics:")
    print(f"  Data type: {df[column_name].dtype}")
    print(f"  Total values: {len(df[column_name])}")
    print(f"  Non-null values: {df[column_name].count()}")
    print(f"  Null values: {df[column_name].isnull().sum()}")
    print(f"  Min value: {df[column_name].min()}")
    print(f"  Max value: {df[column_name].max()}")
    print(f"  Mean value: {df[column_name].mean()}")
    print(f"  Sample values: {df[column_name].head(10).tolist()}")
    
    # If all values are NaN, investigate further
    if df[column_name].isnull().all():
        print(f"ERROR: All values in column '{column_name}' are NaN!")
        print("This could be due to:")
        print("1. Column mapping issue")
        print("2. Data loading problem")
        print("3. Data format incompatibility")
        
        # Show all available data
        print("\nAll available columns and sample data:")
        for col in df.columns:
            print(f"  {col}: {df[col].head(3).tolist()}")
        return
    
    # Plot raw time series
    fig = plot_xrs_time_series(df, column_name, 
                              title=f'GOES XRS {args.channel} Raw Data', log_scale=True)
    fig.savefig(os.path.join(args.output, f'raw_timeseries_{args.channel}.png'), 
               dpi=settings.VISUALIZATION_PARAMS['dpi'])
    
    # Add after loading data
    print(f"Data shape: {df.shape}")
    print(f"Data range - Min: {df[column_name].min():.2e}, Max: {df[column_name].max():.2e}")
    print(f"Data statistics:\n{df[column_name].describe()}")
    
    # Check for any actual flux variations
    flux_std = df[column_name].std()
    flux_mean = df[column_name].mean()
    print(f"Flux variation - Mean: {flux_mean:.2e}, Std: {flux_std:.2e}")
    print(f"Signal-to-noise ratio: {flux_mean/flux_std:.2f}")
    
    # Use the same column name for detection (fix the inconsistency)
    flux_col = column_name  # Use the same column name that we validated above
      # Step 2: Detect flares with traditional method
    print("\n=== Detecting Flares with Traditional Method ===")
    # flux_col is already set above to be the same as column_name
    
    # Debug: Show detection parameters
    print(f"Detection parameters:")
    print(f"  Threshold factor: {settings.DETECTION_PARAMS['threshold_factor']}")
    print(f"  Window size: {settings.DETECTION_PARAMS['window_size']}")
    print(f"  Start threshold: {settings.DETECTION_PARAMS['start_threshold']}")
    print(f"  End threshold: {settings.DETECTION_PARAMS['end_threshold']}")
      # Detect peaks
    print("Detecting peaks...")
    
    # Try more sensitive detection first
    sensitive_params = {
        'threshold_factor': 1.5,  # Lower threshold for more sensitivity
        'window_size': 11,        # Ensure odd number
    }
    
    print(f"Trying sensitive detection with threshold_factor={sensitive_params['threshold_factor']}")
    peaks = detect_flare_peaks(
        df, flux_col,
        threshold_factor=sensitive_params['threshold_factor'],
        window_size=sensitive_params['window_size']
    )
    print(f"Detected {len(peaks)} potential flare peaks with sensitive parameters")
    
    # If no peaks found, try even more sensitive
    if len(peaks) == 0:
        print("No peaks found with sensitive parameters, trying very sensitive...")
        very_sensitive_params = {
            'threshold_factor': 1.0,  # Very low threshold
            'window_size': 7,         # Smaller window
        }
        peaks = detect_flare_peaks(
            df, flux_col,
            threshold_factor=very_sensitive_params['threshold_factor'],
            window_size=very_sensitive_params['window_size']
        )
        print(f"Detected {len(peaks)} potential flare peaks with very sensitive parameters")
      # Define flare bounds
    print("Defining flare bounds...")
    flares = define_flare_bounds(
        df, flux_col, peaks['peak_index'].values,
        start_threshold=settings.DETECTION_PARAMS['start_threshold'],
        end_threshold=settings.DETECTION_PARAMS['end_threshold'],
        min_duration=settings.DETECTION_PARAMS['min_duration'],
        max_duration=settings.DETECTION_PARAMS['max_duration']
    )
    print(f"Defined bounds for {len(flares)} flares")
    print(f"Flares DataFrame columns: {flares.columns.tolist()}")
    print(f"Flares DataFrame shape: {flares.shape}")
    if len(flares) > 0:
        print(f"First flare sample: {flares.iloc[0].to_dict()}")
    
    # Plot detected flares
    fig = plot_detected_flares(df, flux_col, flares)
    fig.savefig(os.path.join(args.output, f'detected_flares_{args.channel}.png'), 
               dpi=settings.VISUALIZATION_PARAMS['dpi'])
    
    # Step 3: Detect overlapping flares
    print("\n=== Detecting Overlapping Flares ===")
    overlapping = detect_overlapping_flares(flares, min_overlap='2min')
    print(f"Detected {len(overlapping)} potentially overlapping flare pairs")
    
    if overlapping:
        # Print information about overlapping flares
        print("Overlapping flare pairs:")
        for i, j, duration in overlapping:
            print(f"  Flares {i+1} and {j+1} overlap by {duration}")
    
    # Step 4: Remove background
    print("\n=== Removing Background Flux ===")
    df_bg = remove_background(
        df, 
        window_size=settings.BACKGROUND_PARAMS['window_size'],
        quantile=settings.BACKGROUND_PARAMS['quantile']
    )
    
    # Plot background-subtracted time series
    fig = plot_xrs_time_series(df_bg, f'{flux_col}_no_background', 
                             title=f'GOES XRS {args.channel} Background-Subtracted Data',
                             log_scale=True)
    fig.savefig(os.path.join(args.output, f'background_subtracted_{args.channel}.png'),
               dpi=settings.VISUALIZATION_PARAMS['dpi'])
    
    # Step 5: Load ML model for flare separation
    print("\n=== Loading ML Model for Flare Separation ===")
    model = FlareDecompositionModel(
        sequence_length=settings.ML_PARAMS['sequence_length'],
        n_features=settings.ML_PARAMS['n_features'],
        max_flares=settings.ML_PARAMS['max_flares']
    )
    model.build_model()
    
    try:
        model.load_model(args.model + '.h5')
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Training a new model with synthetic data...")
        model = train_model(args)
    
    # Step 6: Apply ML model to separate overlapping flares
    print("\n=== Separating Overlapping Flares with ML Model ===")
    
    # Prepare data for ML analysis - focus on regions with overlapping flares
    ml_results = []
    
    if overlapping:
        for i, j, duration in overlapping:
            # Get the start and end indices for the overlapping region
            start_idx = min(flares.iloc[i]['start_index'], flares.iloc[j]['start_index'])
            end_idx = max(flares.iloc[i]['end_index'], flares.iloc[j]['end_index'])
            
            # Ensure we have enough context around the flares
            padding = settings.ML_PARAMS['sequence_length'] // 4
            start_idx = max(0, start_idx - padding)
            end_idx = min(len(df) - 1, end_idx + padding)
            
            # Extract the time series segment
            segment = df.iloc[start_idx:end_idx][flux_col].values
            
            # Ensure the segment has the required length for the model
            if len(segment) < settings.ML_PARAMS['sequence_length']:
                # Pad if too short
                segment = np.pad(segment, 
                                (0, settings.ML_PARAMS['sequence_length'] - len(segment)), 
                                'constant')
            elif len(segment) > settings.ML_PARAMS['sequence_length']:
                # Truncate or use a sliding window if too long
                segment = segment[:settings.ML_PARAMS['sequence_length']]
            
            # Reshape for model input
            segment = segment.reshape(1, -1, 1)
            
            # Decompose the flares
            original, individual_flares, combined = reconstruct_flares(
                model, segment, window_size=settings.ML_PARAMS['sequence_length'], plot=False
            )
            
            # Store the result
            ml_results.append({
                'overlapping_pair': (i, j),
                'start_idx': start_idx,
                'end_idx': end_idx,
                'original': original.flatten(),
                'individual_flares': individual_flares,
                'combined': combined.flatten()
            })
            
            # Plot the decomposition
            timestamps = df.index[start_idx:start_idx+len(segment.flatten())]
            fig = plot_flare_decomposition(original.flatten(), individual_flares, timestamps)
            fig.savefig(os.path.join(args.output, f'decomposed_flares_{i+1}_{j+1}.png'),
                       dpi=settings.VISUALIZATION_PARAMS['dpi'])
    
    # Step 7: Calculate flare energies
    print("\n=== Calculating Flare Energies ===")
    
    # Traditional method - no separation
    print("Calculating energies with traditional method...")
    energy_results_trad = {}
    
    for i, flare in flares.iterrows():
        start_idx = flare['start_index']
        end_idx = flare['end_index']
        
        # Extract the flare segment
        flare_segment = df.iloc[start_idx:end_idx+1].copy()
        
        # Calculate energy
        flare_energy = calculate_flare_energy(
            flare_segment, flux_col, 
            background_column=f'{flux_col}_background' if f'{flux_col}_background' in df_bg.columns else None
        )
        
        # Store the energy
        energy_results_trad[i] = {
            'peak_flux': flare['peak_flux'],
            'integrated_flux': flare_energy['energy'].iloc[-1] if 'energy' in flare_energy.columns else None,
            'duration': flare['duration']
        }
    
    # ML method - with separation
    print("Calculating energies with ML separation...")
    energy_results_ml = {}
    
    for result in ml_results:
        i, j = result['overlapping_pair']
        individual_flares = result['individual_flares']
        
        # For each separated flare
        for k in range(individual_flares.shape[1]):
            if np.max(individual_flares[:, k]) > 0.05 * np.max(result['original']):
                # Calculate the energy using trapezoidal rule
                energy = np.trapezoid(individual_flares[:, k])
                
                # Store the energy
                energy_results_ml[f"{i}_{j}_{k}"] = {
                    'peak_flux': np.max(individual_flares[:, k]),
                    'integrated_flux': energy,
                    'original_flare': (i, j)
                }
    
    # Step 8: Fit power-law distributions
    print("\n=== Fitting Power-Law Distributions ===")
    
    # Prepare energy lists
    traditional_energies = [result['integrated_flux'] for result in energy_results_trad.values() 
                           if result['integrated_flux'] is not None]
    ml_energies = [result['integrated_flux'] for result in energy_results_ml.values() 
                   if result['integrated_flux'] is not None]
    
    # Fit power laws
    print("Fitting power law to traditional method energies...")
    powerlaw_trad = fit_power_law(
        traditional_energies,
        xmin=settings.POWERLAW_PARAMS['xmin'],
        xmax=settings.POWERLAW_PARAMS['xmax'],
        n_bootstrap=settings.POWERLAW_PARAMS['n_bootstrap'],
        plot=True
    )
    plt.savefig(os.path.join(args.output, 'powerlaw_traditional.png'), 
               dpi=settings.VISUALIZATION_PARAMS['dpi'])
    
    if ml_energies:
        print("Fitting power law to ML-separated energies...")
        powerlaw_ml = fit_power_law(
            ml_energies,
            xmin=settings.POWERLAW_PARAMS['xmin'],
            xmax=settings.POWERLAW_PARAMS['xmax'],
            n_bootstrap=settings.POWERLAW_PARAMS['n_bootstrap'],
            plot=True
        )
        plt.savefig(os.path.join(args.output, 'powerlaw_ml_separated.png'),
                   dpi=settings.VISUALIZATION_PARAMS['dpi'])
        
        # Compare the power laws
        print("\n=== Comparing Power-Law Distributions ===")
        comparison = compare_flare_populations(
            traditional_energies, "Traditional Method",
            ml_energies, "ML-Separated",
            xmin=settings.POWERLAW_PARAMS['xmin'],
            xmax=settings.POWERLAW_PARAMS['xmax'],
            plot=True
        )
        plt.savefig(os.path.join(args.output, 'powerlaw_comparison.png'),
                   dpi=settings.VISUALIZATION_PARAMS['dpi'])
        
        # Print comparison results
        print("\nPower-law comparison results:")
        print(f"  Traditional method: α = {powerlaw_trad['alpha']:.3f} ± {powerlaw_trad['alpha_err']:.3f}")
        print(f"  ML-separated method: α = {powerlaw_ml['alpha']:.3f} ± {powerlaw_ml['alpha_err']:.3f}")
        print(f"  Difference: {comparison['alpha_diff']:.3f} ± {comparison['alpha_err_combined']:.3f}")
        print(f"  Significance: {comparison['significance']:.2f}σ")
        print(f"  p-value: {comparison['p_value']:.3e}")
    else:
        print("No ML-separated flare energies available for comparison.")
    
    print("\n=== Analysis Complete ===")
    print(f"Results saved to {args.output}")


def train_bayesian_model(args):
    """Train the enhanced Bayesian flare analysis model."""
    print("\n=== Training Enhanced Bayesian Flare Analysis Model ===")
    
    # Initialize enhanced Bayesian analyzer
    analyzer = BayesianFlareAnalyzer(
        sequence_length=128,
        n_features=2,
        max_flares=3,
        n_monte_carlo_samples=100,
        sensor_noise_std=0.01
    )
    
    print("Building enhanced Bayesian neural network...")
    model = analyzer.build_bayesian_model()
    print(f"Model built with {model.count_params():,} parameters")
    
    # Generate physics-based synthetic data
    print("Generating synthetic solar flare data with physics constraints...")
    X_synthetic, y_synthetic = analyzer.generate_synthetic_data_with_physics(
        n_samples=2000, noise_level=0.02
    )
    print(f"Generated {X_synthetic.shape[0]} synthetic samples")
    
    # Train with Monte Carlo data augmentation
    print("Training with Monte Carlo data augmentation...")
    history = analyzer.train_bayesian_model(
        X_synthetic, y_synthetic,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        augment_data=True
    )
    
    # Save the trained model
    model_path = os.path.join(args.output, 'enhanced_bayesian_flare_model')
    analyzer.save_bayesian_model(model_path)
    print(f"Enhanced Bayesian model saved to {model_path}")
    
    # Demonstrate uncertainty quantification
    print("\nDemonstrating uncertainty quantification...")
    X_test = X_synthetic[-100:]
    y_test = y_synthetic[-100:]
    
    # Monte Carlo predictions
    predictions = analyzer.monte_carlo_predict(X_test, n_samples=100)
    print(f"Mean uncertainty: {np.mean(predictions['std']):.6f}")
    
    # Separate epistemic and aleatoric uncertainty
    uncertainty_components = analyzer.quantify_epistemic_aleatoric_uncertainty(
        X_test[:20], n_epistemic=30, n_aleatoric=50
    )
    
    print(f"Epistemic uncertainty: {np.mean(uncertainty_components['epistemic_uncertainty']):.6f}")
    print(f"Aleatoric uncertainty: {np.mean(uncertainty_components['aleatoric_uncertainty']):.6f}")
    
    # MCMC sampling
    try:
        print("Performing MCMC sampling...")
        mcmc_results = analyzer.mcmc_sampling(
            X_test[:10], y_test[:10],
            n_samples=200, n_burnin=100
        )
        print(f"MCMC acceptance rate: {mcmc_results['acceptance_rate']:.3f}")
    except Exception as e:
        print(f"MCMC sampling issue: {e}")
    
    # Visualization
    fig = analyzer.plot_uncertainty_analysis(X_test, predictions, y_test)
    plt.savefig(os.path.join(args.output, 'bayesian_uncertainty_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Enhanced Bayesian model training completed!")
    return analyzer


def main():
    """Enhanced main entry point with comprehensive analysis options."""
    print("Enhanced Solar Flare Analysis Pipeline")
    print("=====================================")
    
    # Parse command line arguments
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    try:
        # Generate synthetic data if requested
        if args.generate_synthetic:
            generate_synthetic_data(args)
            return
        
        # Run comprehensive analysis if requested
        if args.comprehensive:
            print("Running comprehensive analysis with all features...")
            run_comprehensive_analysis(args)
            return
        
        # Train enhanced model if requested
        if args.train_enhanced:
            if ENHANCED_MODELS_AVAILABLE:
                train_enhanced_model(args)
            else:
                print("Enhanced models not available - falling back to basic training")
                train_model(args)
            
            # Run analysis after training if data is provided
            if args.data:
                print("\nRunning analysis with newly trained model...")
                run_comprehensive_analysis(args)
            return
        
        # Train basic model if requested
        if args.train:
            train_model(args)
            
            # Run analysis after training if data is provided
            if args.data:
                print("\nRunning analysis with newly trained model...")
                analyze_flares(args)
            return
        
        # Train Bayesian model if requested
        if args.train_bayesian:
            train_bayesian_model(args)
            return
        
        # Run specific analysis types
        if args.nanoflare_analysis or args.corona_heating:
            run_comprehensive_analysis(args)
            return
        
        # Default: run basic flare analysis
        if args.data:
            analyze_flares(args)
        else:
            print("No analysis specified. Use --help for options.")
            print("\nAvailable analysis modes:")
            print("  --generate-synthetic : Generate synthetic data and save to CSV")
            print("  --comprehensive      : Full analysis with all features")
            print("  --train-enhanced     : Train enhanced ML model")
            print("  --nanoflare-analysis : Focus on nanoflare detection")
            print("  --corona-heating     : Assess corona heating contribution")
            print("  --train              : Train basic ML model")
            print("  --data <path>        : Analyze specific data file/directory")
    
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\nOutput directory: {args.output}")
        print("Analysis complete!")


if __name__ == "__main__":
    main()

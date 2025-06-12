#!/usr/bin/env python3
"""
Fixed Production-Level Training Script for Solar Flare Analysis ML Models
Trains all built-in models with proper method signatures and error handling
"""

import sys
import os
import logging
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class XRSDataLoader:
    """Enhanced XRS data loader with robust column mapping and data cleaning"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.data = None
        self.processed_data = None
        
    def load_all_data(self, max_files=None, sample_rate=0.3):
        """Load XRS CSV files with enhanced column mapping and data cleaning"""
        logger.info(f"Loading XRS data from: {self.data_dir}")
        
        csv_files = list(self.data_dir.glob("*.csv"))
        if max_files:
            csv_files = csv_files[:max_files]
            
        logger.info(f"Found {len(csv_files)} CSV files")
        
        all_data = []
        total_samples = 0
        
        for csv_file in csv_files:
            logger.info(f"Processing: {csv_file.name}")
            try:
                # Load CSV data
                df = pd.read_csv(csv_file)
                
                # Sample data for memory efficiency if file is large
                if len(df) > 50000:
                    df = df.sample(frac=sample_rate, random_state=42)
                    logger.info(f"   Sampled to {len(df)} rows for memory efficiency")
                
                # Standardize column names with enhanced mapping
                df_clean = self._standardize_columns(df)
                
                if df_clean is not None and len(df_clean) > 0:
                    # Log transform and handle zeros
                    df_clean['xrsa_log'] = np.log10(np.maximum(df_clean['xrs_a'], 1e-10))
                    df_clean['xrsb_log'] = np.log10(np.maximum(df_clean['xrs_b'], 1e-10))
                    
                    # Store both raw and log data
                    flux_data = df_clean[['xrsa_log', 'xrsb_log']].values
                    all_data.append(flux_data)
                    total_samples += len(df_clean)
                    logger.info(f"   Added {len(df_clean)} valid samples")
                else:
                    logger.warning(f"No valid data found in {csv_file.name}")
                    
            except Exception as e:
                logger.error(f"Error loading {csv_file.name}: {e}")
                continue
                
        if all_data:
            self.data = np.vstack(all_data)
            logger.info(f"Total combined samples: {total_samples}")
        else:
            logger.error("No data loaded successfully")
            self.data = np.array([])
            
        return self.data
    
    def _standardize_columns(self, df):
        """Enhanced column standardization for various XRS CSV formats"""
        # Comprehensive XRS column mappings
        column_mappings = {
            # Format 1: xrsa_flux_observed, xrsb_flux_observed
            'xrsa_flux_observed': 'xrs_a',
            'xrsb_flux_observed': 'xrs_b',
            # Format 2: xrsa_flux, xrsb_flux
            'xrsa_flux': 'xrs_a',
            'xrsb_flux': 'xrs_b',
            # Format 3: xrs_a, xrs_b (already standard)
            'xrs_a': 'xrs_a',
            'xrs_b': 'xrs_b',
            # Format 4: XRSA, XRSB (uppercase)
            'XRSA': 'xrs_a',
            'XRSB': 'xrs_b',
            # Format 5: xrs-a, xrs-b (with hyphens)
            'xrs-a': 'xrs_a',
            'xrs-b': 'xrs_b',
            # Time columns
            'time_minutes': 'time_minutes',
            'time_seconds': 'time_seconds',
            'datetime': 'datetime',
            'timestamp': 'datetime',
            'time': 'datetime'
        }
        
        # Apply column mappings
        df_renamed = df.rename(columns=column_mappings)
        
        # Check for required XRS channels
        if 'xrs_a' not in df_renamed.columns or 'xrs_b' not in df_renamed.columns:
            logger.warning(f"Missing XRS channels. Available columns: {df.columns.tolist()}")
            return None
        
        # Clean data - remove NaN values
        df_clean = df_renamed.dropna(subset=['xrs_a', 'xrs_b'])
        
        # Remove invalid flux values (negative, zero, or unreasonably large)
        original_len = len(df_clean)
        df_clean = df_clean[
            (df_clean['xrs_a'] > 0) & 
            (df_clean['xrs_b'] > 0) &
            (df_clean['xrs_a'] < 1e-2) &  # Upper limit for reasonable XRS values
            (df_clean['xrs_b'] < 1e-2) &
            (df_clean['xrs_a'] > 1e-12) &  # Lower limit for reasonable XRS values
            (df_clean['xrs_b'] > 1e-12)
        ]
        
        if len(df_clean) < original_len * 0.1:  # If we lost more than 90% of data
            logger.warning(f"Heavy data filtering: {original_len} -> {len(df_clean)} samples")
        
        # Create time index if missing
        if 'datetime' not in df_clean.columns and len(df_clean) > 0:
            df_clean = df_clean.copy()
            df_clean['datetime'] = pd.date_range(
                start='2000-01-01', 
                periods=len(df_clean), 
                freq='1min'
            )
        
        return df_clean
        
    def create_sequences(self, sequence_length=128, overlap=0.5):
        """Create overlapping sequences with configurable overlap"""
        if self.data is None or len(self.data) == 0:
            logger.warning("No data available for sequence creation")
            return np.array([])
            
        step_size = int(sequence_length * (1 - overlap))
        sequences = []
        n_samples = len(self.data)
        
        for i in range(0, n_samples - sequence_length + 1, step_size):
            seq = self.data[i:i + sequence_length]
            sequences.append(seq)
            
        logger.info(f"Created {len(sequences)} sequences of length {sequence_length} with {overlap*100}% overlap")
        return np.array(sequences)

class ProductionMLTrainer:
    """Production-level trainer for all ML models"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.output_dir = Path("output")
        self.models_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        self.training_results = {}
        
    def train_bayesian_flare_analyzer(self, data):
        """Train Bayesian Flare Analyzer with correct method calls"""
        logger.info("Training Bayesian Flare Analyzer...")
        
        try:
            from src.ml_models.bayesian_flare_analysis import BayesianFlareAnalyzer
            
            # Initialize analyzer
            analyzer = BayesianFlareAnalyzer(
                sequence_length=128,
                n_features=2,
                max_flares=3,
                n_monte_carlo_samples=50,
                sensor_noise_std=0.01
            )
            
            # Build model
            logger.info("Building Bayesian neural network...")
            model = analyzer.build_bayesian_model()
            logger.info(f"Model built with {model.count_params():,} parameters")
            
            # Generate synthetic data for training
            logger.info("Generating physics-based synthetic data...")
            X_synthetic, y_synthetic = analyzer.generate_synthetic_data_with_physics(
                n_samples=2000, noise_level=0.02
            )
            
            # Train model
            logger.info("Training with Monte Carlo data augmentation...")
            history = analyzer.train_bayesian_model(
                X_synthetic, y_synthetic,
                validation_split=0.2,
                epochs=30,
                batch_size=32,
                augment_data=True
            )
            
            # Test uncertainty quantification
            logger.info("Testing uncertainty quantification...")
            X_test = X_synthetic[-50:]
            
            # Use monte_carlo_inference instead of analyze
            inference_results = analyzer.monte_carlo_inference(
                X_test, n_samples=100, chains=2
            )
            
            # Save model
            model_path = self.models_dir / 'bayesian_flare_analyzer.h5'
            model.save(str(model_path))
              # Create visualizations
            self._plot_bayesian_results(inference_results, X_test)
            
            return {
                'model': analyzer,
                'history': history,
                'uncertainty_results': inference_results,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error training Bayesian Flare Analyzer: {e}")
            return {'error': str(e), 'status': 'failed'}
            
    def train_enhanced_flare_models(self, data):
        """Train Enhanced Flare Analysis models with improved error handling"""
        logger.info("Training Enhanced Flare Analysis models...")
        
        try:
            from src.ml_models.enhanced_flare_analysis import (
                NanoflareDetector, FlareEnergyAnalyzer, EnhancedFlareDecompositionModel
            )
            
            results = {}
            
            # Validate input data
            if len(data) == 0:
                raise ValueError("No training data provided")
            
            # Train NanoflareDetector with improved parameters
            logger.info("Training NanoflareDetector...")
            nanoflare_detector = NanoflareDetector(
                min_energy_threshold=1e-9,
                alpha_threshold=2.0
            )
            
            # Generate test data for nanoflare detection based on actual data statistics
            data_std = np.std(data) if len(data) > 0 else 1e-8
            data_mean = np.mean(data) if len(data) > 0 else 1e-7
            
            # Use log-normal distribution based on data characteristics
            test_energies = np.random.lognormal(
                mean=np.log(max(data_mean, 1e-10)), 
                sigma=max(data_std, 0.1), 
                size=min(1000, len(data))
            )
            
            nanoflare_results = nanoflare_detector.detect_nanoflares(test_energies)
            results['nanoflare_detector'] = nanoflare_results
            
            # Train FlareEnergyAnalyzer
            logger.info("Training FlareEnergyAnalyzer...")
            energy_analyzer = FlareEnergyAnalyzer()
            
            # Use actual data for energy analysis instead of purely synthetic
            if len(data) > 500:
                sample_data = data[np.random.choice(len(data), 500, replace=False)]
            else:
                sample_data = data
                
            # Convert log flux data back to linear for energy analysis
            linear_flares = np.power(10, sample_data.flatten())
            energy_analysis = energy_analyzer.analyze_energy_distribution(linear_flares)
            results['energy_analyzer'] = energy_analysis
            
            # Train EnhancedFlareDecompositionModel with validation
            logger.info("Training EnhancedFlareDecompositionModel...")
            enhanced_model = EnhancedFlareDecompositionModel(
                sequence_length=min(256, len(data) // 10),  # Adaptive sequence length
                n_features=2,
                max_flares=5,
                dropout_rate=0.3
            )
            
            # Build model with error handling
            try:
                enhanced_model.build_enhanced_model()
                logger.info(f"Enhanced model built successfully")
            except Exception as model_error:
                logger.error(f"Error building enhanced model: {model_error}")
                # Continue with simpler model if enhanced fails
                enhanced_model = EnhancedFlareDecompositionModel(
                    sequence_length=128,
                    n_features=2,
                    max_flares=3,
                    dropout_rate=0.2
                )
                enhanced_model.build_enhanced_model()
            
            # Generate enhanced synthetic data with better parameters
            X_enhanced, y_enhanced = enhanced_model.generate_enhanced_synthetic_data(
                n_samples=min(1500, max(500, len(data) // 2)),  # Adaptive sample size
                noise_level=0.05
            )
            
            # Train enhanced model with adaptive parameters
            batch_size = min(32, max(8, len(X_enhanced) // 50))
            epochs = min(50, max(20, 100000 // len(X_enhanced)))  # Adaptive epochs
            
            history = enhanced_model.train_enhanced_model(
                X_enhanced, y_enhanced,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                patience=max(10, epochs // 5)  # Adaptive patience
            )
            
            results['enhanced_decomposition'] = {
                'model': enhanced_model,
                'history': history,
                'training_samples': len(X_enhanced),
                'epochs_run': epochs,
                'final_loss': history.history['loss'][-1] if history and 'loss' in history.history else None
            }
            
            # Save models with error handling
            try:
                pickle.dump(nanoflare_detector, open(self.models_dir / 'nanoflare_detector.pkl', 'wb'))
                logger.info("Nanoflare detector saved successfully")
            except Exception as e:
                logger.warning(f"Could not save nanoflare detector: {e}")
            
            try:
                pickle.dump(energy_analyzer, open(self.models_dir / 'energy_analyzer.pkl', 'wb'))
                logger.info("Energy analyzer saved successfully")
            except Exception as e:
                logger.warning(f"Could not save energy analyzer: {e}")
            
            try:
                enhanced_model.model.save(str(self.models_dir / 'enhanced_decomposition.h5'))
                logger.info("Enhanced decomposition model saved successfully")
            except Exception as e:
                logger.warning(f"Could not save enhanced model: {e}")
            
            # Create visualizations with error handling
            try:
                self._plot_enhanced_results(results)
            except Exception as e:
                logger.warning(f"Could not create enhanced results plots: {e}")
            
            return {
                'models': results,
                'status': 'success',
                'data_samples_used': len(data),
                'models_trained': len(results)
            }
            
        except Exception as e:
            logger.error(f"Error training Enhanced Flare models: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'error': str(e), 'status': 'failed'}
            
    def train_flare_decomposition_model(self, data):
        """Train Flare Decomposition Model with correct method signature"""
        logger.info("Training Flare Decomposition Model...")
        
        try:
            from src.ml_models.flare_decomposition import FlareDecompositionModel
            
            # Initialize model
            model = FlareDecompositionModel(
                sequence_length=128,
                n_features=2,
                max_flares=3
            )
            
            model.build_model()
            logger.info(f"Model built with {model.model.count_params():,} parameters")
            
            # Generate training data
            logger.info("Generating synthetic training data...")
            X_train, y_train = model.generate_synthetic_data(n_samples=3000)
            X_val, y_val = model.generate_synthetic_data(n_samples=500)
            
            # Train model (fix method call - no validation_data parameter)
            history = model.train(
                X_train, y_train,
                validation_split=0.2,  # Use validation_split instead
                epochs=30,
                batch_size=32,
                save_path=str(self.models_dir / 'flare_decomposition.h5')
            )
            
            # Evaluate model
            logger.info("Evaluating model...")
            eval_results = model.evaluate(X_val, y_val)
            
            # Save model
            model.save_model(str(self.models_dir / 'flare_decomposition_full.h5'))
            
            # Create visualizations
            self._plot_decomposition_results(history, eval_results)
            
            return {
                'model': model,
                'history': history,
                'evaluation': eval_results,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error training Flare Decomposition Model: {e}")
            return {'error': str(e), 'status': 'failed'}
            
    def train_monte_carlo_simulator(self, data):
        """Train Monte Carlo Background Simulator"""
        logger.info("Training Monte Carlo Background Simulator...")
        
        try:
            from src.ml_models.monte_carlo_background_simulation import MonteCarloBackgroundSimulator
            
            # Initialize simulator
            simulator = MonteCarloBackgroundSimulator()
            
            # Load models without MSE metric issue
            logger.info("Initializing simulation environment...")
            try:
                simulator.load_models()
            except Exception as model_load_error:
                logger.warning(f"Could not load pre-trained models: {model_load_error}")
                logger.info("Continuing with simulation-only training...")
            
            # Generate background data
            logger.info("Generating background simulations...")
            background_data = simulator.generate_background_data(
                n_samples=1000,
                duration_hours=24
            )
            
            # Run Monte Carlo simulations
            logger.info("Running Monte Carlo simulations...")
            simulation_results = simulator.run_monte_carlo_simulation(
                background_data,
                n_iterations=100,
                add_flare_events=True
            )
            
            # Save simulation results
            with open(self.models_dir / 'monte_carlo_results.json', 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = {}
                for key, value in simulation_results.items():
                    if isinstance(value, np.ndarray):
                        serializable_results[key] = value.tolist()
                    else:
                        serializable_results[key] = value
                json.dump(serializable_results, f, indent=2)
            
            # Create visualizations
            self._plot_monte_carlo_results(simulation_results)
            
            return {
                'simulator': simulator,
                'simulation_results': simulation_results,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error training Monte Carlo Simulator: {e}")
            return {'error': str(e), 'status': 'failed'}
            
    def _plot_bayesian_results(self, results, test_data):
        """Create visualizations for Bayesian analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Bayesian Flare Analysis Results', fontsize=16)
        
        # Plot uncertainty metrics
        uncertainty = results.get('uncertainty_metrics', {})
        
        axes[0, 0].hist(results.get('std', []), bins=20, alpha=0.7)
        axes[0, 0].set_title('Prediction Uncertainty Distribution')
        axes[0, 0].set_xlabel('Standard Deviation')
        axes[0, 0].set_ylabel('Frequency')
        
        # Plot sample predictions with uncertainty
        mean_pred = results.get('mean', np.array([]))
        if len(mean_pred) > 0:
            x_range = np.arange(len(mean_pred))
            axes[0, 1].plot(x_range, mean_pred, 'b-', label='Mean Prediction')
            
            std_pred = results.get('std', np.zeros_like(mean_pred))
            axes[0, 1].fill_between(x_range, 
                                  mean_pred - std_pred,
                                  mean_pred + std_pred,
                                  alpha=0.3, label='Uncertainty')
            axes[0, 1].set_title('Predictions with Uncertainty')
            axes[0, 1].legend()
        
        # Plot confidence intervals
        ci = results.get('confidence_intervals', {})
        if '2.5th' in ci and '97.5th' in ci:
            axes[1, 0].fill_between(x_range, ci['2.5th'].flatten(), ci['97.5th'].flatten(),
                                  alpha=0.3, label='95% CI')
            axes[1, 0].plot(x_range, mean_pred, 'r-', label='Mean')
            axes[1, 0].set_title('Confidence Intervals')
            axes[1, 0].legend()
        
        # Plot sample time series
        if len(test_data) > 0:
            sample_idx = 0
            axes[1, 1].plot(test_data[sample_idx, :, 0], label='XRS-A')
            if test_data.shape[2] > 1:
                axes[1, 1].plot(test_data[sample_idx, :, 1], label='XRS-B')
            axes[1, 1].set_title('Sample Input Time Series')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'bayesian_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_enhanced_results(self, results):
        """Create visualizations for enhanced flare analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Enhanced Flare Analysis Results', fontsize=16)
        
        # Plot nanoflare detection
        nanoflare_data = results.get('nanoflare_detector', {})
        if 'nanoflare_count' in nanoflare_data:
            categories = ['Nanoflares', 'Regular Flares']
            counts = [nanoflare_data.get('nanoflare_count', 0),
                     nanoflare_data.get('regular_flare_count', 0)]
            axes[0, 0].bar(categories, counts)
            axes[0, 0].set_title('Flare Detection Results')
            axes[0, 0].set_ylabel('Count')
        
        # Plot energy distribution
        energy_data = results.get('energy_analyzer', {})
        if 'mean_energy' in energy_data:
            stats = ['Mean', 'Median', 'Std', 'Max']
            values = [
                energy_data.get('mean_energy', 0),
                energy_data.get('median_energy', 0),
                energy_data.get('std_energy', 0),
                energy_data.get('max_energy', 0)
            ]
            axes[0, 1].bar(stats, np.log10(np.maximum(values, 1e-10)))
            axes[0, 1].set_title('Energy Statistics (log10)')
            axes[0, 1].set_ylabel('Log10(Energy)')
        
        # Plot training history if available
        enhanced_decomp = results.get('enhanced_decomposition', {})
        history = enhanced_decomp.get('history')
        if history and hasattr(history, 'history'):
            axes[1, 0].plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                axes[1, 0].plot(history.history['val_loss'], label='Validation Loss')
            axes[1, 0].set_title('Enhanced Model Training History')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
        
        # Plot power law analysis if available
        power_law = energy_data.get('power_law_analysis', {})
        if 'alpha' in power_law and power_law['alpha'] is not None:
            axes[1, 1].text(0.1, 0.5, f"Power Law Index: {power_law['alpha']:.2f}\n"
                                      f"R-squared: {power_law.get('r_squared', 'N/A'):.3f}",
                           transform=axes[1, 1].transAxes, fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='wheat'))
            axes[1, 1].set_title('Power Law Analysis')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'enhanced_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_decomposition_results(self, history, evaluation):
        """Create visualizations for decomposition model"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Flare Decomposition Model Results', fontsize=16)
        
        # Plot training history
        if history and hasattr(history, 'history'):
            axes[0, 0].plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
            axes[0, 0].set_title('Training History')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
        
        # Plot evaluation metrics
        if evaluation:
            metrics = ['MSE', 'R²']
            values = [evaluation.get('mse', 0), evaluation.get('r2', 0)]
            axes[0, 1].bar(metrics, values)
            axes[0, 1].set_title('Evaluation Metrics')
            axes[0, 1].set_ylabel('Score')
        
        # Plot flare-wise metrics
        if 'flare_mse' in evaluation:
            flare_indices = range(len(evaluation['flare_mse']))
            axes[1, 0].bar(flare_indices, evaluation['flare_mse'])
            axes[1, 0].set_title('Per-Flare MSE')
            axes[1, 0].set_xlabel('Flare Index')
            axes[1, 0].set_ylabel('MSE')
        
        if 'flare_r2' in evaluation:
            axes[1, 1].bar(flare_indices, evaluation['flare_r2'])
            axes[1, 1].set_title('Per-Flare R²')
            axes[1, 1].set_xlabel('Flare Index')
            axes[1, 1].set_ylabel('R² Score')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'decomposition_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_monte_carlo_results(self, results):
        """Create visualizations for Monte Carlo simulation"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Monte Carlo Simulation Results', fontsize=16)
        
        # Plot background data
        background_data = results.get('background_data', [])
        if len(background_data) > 0:
            bg_array = np.array(background_data)
            axes[0, 0].plot(bg_array[:500])  # Plot first 500 points
            axes[0, 0].set_title('Background Data Sample')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Flux')
        
        # Plot simulation statistics
        stats = results.get('simulation_statistics', {})
        if stats:
            stat_names = list(stats.keys())[:4]  # First 4 statistics
            stat_values = [stats[name] for name in stat_names]
            axes[0, 1].bar(stat_names, stat_values)
            axes[0, 1].set_title('Simulation Statistics')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot flare events if available
        flare_events = results.get('flare_events', [])
        if flare_events:
            event_times = [event.get('time', i) for i, event in enumerate(flare_events)]
            event_intensities = [event.get('intensity', 1) for event in flare_events]
            axes[1, 0].scatter(event_times, event_intensities, alpha=0.6)
            axes[1, 0].set_title('Simulated Flare Events')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Intensity')
        
        # Plot uncertainty estimates
        uncertainties = results.get('uncertainties', [])
        if uncertainties:
            axes[1, 1].hist(uncertainties, bins=20, alpha=0.7)
            axes[1, 1].set_title('Uncertainty Distribution')
            axes[1, 1].set_xlabel('Uncertainty')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'monte_carlo_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    def train_all_models(self, max_files=5, sample_rate=0.3):
        """Train all models with enhanced data loading and error handling"""
        logger.info("="*50)
        logger.info("Starting comprehensive ML models training")
        logger.info("="*50)
        
        # Load data with enhanced loader
        logger.info("Loading XRS data with enhanced processing...")
        data_loader = XRSDataLoader("data/XRS")
        
        try:
            raw_data = data_loader.load_all_data(max_files=max_files, sample_rate=sample_rate)
            
            if len(raw_data) == 0:
                logger.error("No data loaded. Cannot proceed with training.")
                return {'error': 'No data available', 'status': 'failed'}
                
            logger.info(f"Successfully loaded {len(raw_data)} data points")
            
            # Create sequences with configurable overlap
            sequences = data_loader.create_sequences(sequence_length=128, overlap=0.5)
            logger.info(f"Created {len(sequences)} training sequences")
            
            if len(sequences) == 0:
                logger.warning("No sequences created, using raw data for training")
                sequences = raw_data
                
        except Exception as e:
            logger.error(f"Failed to load and process data: {e}")
            return {'error': f'Data loading failed: {str(e)}', 'status': 'failed'}
        
        results = {}
        
        # Enhanced model training with better error handling
        models_to_train = [
            ("Bayesian Flare Analyzer", self.train_bayesian_flare_analyzer),
            ("Enhanced Flare Models", self.train_enhanced_flare_models),
            ("Flare Decomposition Model", self.train_flare_decomposition_model),
            ("Monte Carlo Simulator", self.train_monte_carlo_simulator)
        ]
        
        successful_models = 0
        failed_models = 0
        
        for model_name, train_method in models_to_train:
            logger.info("")
            logger.info("="*40)
            logger.info(f"Training: {model_name}")
            logger.info("="*40)
            
            try:
                # Add timeout and resource monitoring
                import time
                start_time = time.time()
                
                result = train_method(sequences)
                
                training_time = time.time() - start_time
                result['training_time_seconds'] = training_time
                
                results[model_name] = result
                
                if result.get('status') == 'success':
                    logger.info(f"✓ {model_name} training completed successfully in {training_time:.1f}s")
                    successful_models += 1
                else:
                    logger.error(f"✗ {model_name} training failed: {result.get('error', 'Unknown error')}")
                    failed_models += 1
                    
            except Exception as e:
                error_msg = f"Exception in {model_name}: {str(e)}"
                logger.error(error_msg)
                results[model_name] = {
                    'error': error_msg, 
                    'status': 'failed',
                    'exception_type': type(e).__name__
                }
                failed_models += 1
        
        # Enhanced training summary with more details
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(models_to_train),
            'successful_models': successful_models,
            'failed_models': failed_models,
            'success_rate': successful_models / len(models_to_train) * 100,
            'data_statistics': {
                'raw_data_points': len(raw_data),
                'sequence_samples': len(sequences),
                'data_shape': list(raw_data.shape) if hasattr(raw_data, 'shape') else 'N/A',
                'sequence_shape': list(sequences.shape) if hasattr(sequences, 'shape') else 'N/A'
            },
            'training_parameters': {
                'max_files_processed': max_files,
                'sample_rate': sample_rate,
                'sequence_length': 128,
                'overlap': 0.5
            },
            'results_summary': {
                k: {
                    'status': v.get('status', 'unknown'),
                    'training_time': v.get('training_time_seconds', 0),
                    'error': v.get('error', None) if v.get('status') == 'failed' else None
                } 
                for k, v in results.items()
            }
        }
        
        # Save enhanced summary
        summary_path = self.output_dir / 'training_summary.json'
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Enhanced training summary saved: {summary_path}")
        except Exception as e:
            logger.warning(f"Could not save training summary: {e}")
        
        # Create overall visualization
        try:
            self._create_training_overview(results, summary)
        except Exception as e:
            logger.warning(f"Could not create training overview visualization: {e}")
        
        logger.info("")
        logger.info("="*60)
        logger.info("ENHANCED TRAINING COMPLETED!")
        logger.info("="*60)
        logger.info(f"Success Rate: {successful_models}/{len(models_to_train)} ({summary['success_rate']:.1f}%)")
        logger.info(f"Data Processed: {len(raw_data):,} points -> {len(sequences):,} sequences")
        
        return results

    def _create_training_overview(self, results, summary):
        """Create comprehensive training overview visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ML Models Training Overview', fontsize=16)
        
        # Training status overview
        model_names = list(results.keys())
        statuses = [results[name].get('status', 'unknown') for name in model_names]
        success_count = sum(1 for s in statuses if s == 'success')
        failed_count = sum(1 for s in statuses if s == 'failed')
        
        axes[0, 0].pie([success_count, failed_count], 
                      labels=['Successful', 'Failed'],
                      colors=['lightgreen', 'lightcoral'],
                      autopct='%1.1f%%')
        axes[0, 0].set_title('Training Success Rate')
        
        # Training times
        training_times = [results[name].get('training_time_seconds', 0) for name in model_names]
        if any(t > 0 for t in training_times):
            bars = axes[0, 1].bar(range(len(model_names)), training_times)
            axes[0, 1].set_xticks(range(len(model_names)))
            axes[0, 1].set_xticklabels([name.replace(' ', '\n') for name in model_names], rotation=45)
            axes[0, 1].set_ylabel('Training Time (seconds)')
            axes[0, 1].set_title('Training Duration by Model')
            
            # Color bars by status
            for i, bar in enumerate(bars):
                if statuses[i] == 'success':
                    bar.set_color('lightgreen')
                else:
                    bar.set_color('lightcoral')
        
        # Data processing statistics
        data_stats = summary.get('data_statistics', {})
        stats_labels = ['Raw Data Points', 'Training Sequences']
        stats_values = [
            data_stats.get('raw_data_points', 0),
            data_stats.get('sequence_samples', 0)
        ]
        
        axes[0, 2].bar(stats_labels, stats_values, color='skyblue')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title('Data Processing Summary')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Model-specific results
        for i, (model_name, result) in enumerate(results.items()):
            if i >= 3:  # Only show first 3 models in detail
                break
                
            ax = axes[1, i]
            
            if result.get('status') == 'success':
                # Show success metrics
                metrics = []
                values = []
                
                if 'uncertainty_results' in result:
                    metrics.append('Uncertainty\nSamples')
                    uncertainty = result['uncertainty_results']
                    values.append(len(uncertainty.get('mean', [])))
                
                if 'models' in result:
                    models_trained = len(result['models'])
                    metrics.append('Sub-models\nTrained')
                    values.append(models_trained)
                
                if 'evaluation' in result:
                    eval_data = result['evaluation']
                    if 'mse' in eval_data:
                        metrics.append('MSE\n(×1000)')
                        values.append(eval_data['mse'] * 1000)
                
                if metrics and values:
                    ax.bar(metrics, values, color='lightgreen', alpha=0.7)
                    ax.set_title(f'{model_name.split()[0]}\nSuccess Metrics')
                else:
                    ax.text(0.5, 0.5, '✓ Trained\nSuccessfully', 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=12, color='green',
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
                    ax.set_title(f'{model_name.split()[0]} Status')
            else:
                # Show error information
                error_msg = result.get('error', 'Unknown error')
                ax.text(0.5, 0.5, f'✗ Failed\n{error_msg[:50]}...', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, color='red',
                       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
                ax.set_title(f'{model_name.split()[0]} Status')
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Training overview visualization saved: output/training_overview.png")

    # ...existing plotting methods...
def main():
    """Enhanced main training function with better error handling and reporting"""
    print("Starting enhanced comprehensive ML models training...")
    print("="*60)
    
    try:
        trainer = ProductionMLTrainer()
        
        # Enhanced training with configurable parameters
        results = trainer.train_all_models(
            max_files=5,      # Limit files for testing
            sample_rate=0.3   # Sample rate for large files
        )
        
        logger.info("")
        logger.info("="*60)
        logger.info("ENHANCED TRAINING COMPLETED!")
        logger.info("="*60)
        
        print("="*60)
        print("ENHANCED TRAINING SUMMARY")
        print("="*60)
        
        if not results:
            print("⚠ No results generated - check logs for data loading issues")
            return
        
        successful = 0
        failed = 0
        total_training_time = 0
        
        for model_name, result in results.items():
            training_time = result.get('training_time_seconds', 0)
            total_training_time += training_time
            
            if result.get('status') == 'failed':
                error_msg = result.get('error', 'Unknown error')
                print(f"✗ {model_name}: FAILED - {error_msg[:80]}...")
                failed += 1
            else:
                print(f"✓ {model_name}: SUCCESS ({training_time:.1f}s)")
                successful += 1
        
        success_rate = (successful / len(results)) * 100 if results else 0
        
        print(f"\nOverall Results:")
        print(f"  • {successful} successful, {failed} failed")
        print(f"  • Success rate: {success_rate:.1f}%")
        print(f"  • Total training time: {total_training_time:.1f} seconds")
        print(f"  • Models saved to: {trainer.models_dir}")
        print(f"  • Visualizations saved to: {trainer.output_dir}")
        
        # Additional diagnostic information
        if failed > 0:
            print(f"\n⚠ {failed} models failed - check training.log for detailed error information")
        
        if successful > 0:
            print(f"\n✓ {successful} models trained successfully!")
            print("  Check output/ directory for visualization results")
        
    except Exception as e:
        print(f"Training failed with critical error: {e}")
        logger.error(f"Critical training failure: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        print("Check training.log for detailed error information")


if __name__ == "__main__":
    main()

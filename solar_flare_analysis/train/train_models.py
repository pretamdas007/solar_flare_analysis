#!/usr/bin/env python
"""
Comprehensive training script for solar flare analysis neural network models.
This script will:
1. Load and preprocess GOES XRS data
2. Generate synthetic training data
3. Build and train neural network models
4. Evaluate model performance
5. Save trained models to the models/ directory
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from keras import models, layers, optimizers, callbacks
    from sklearn.model_selection import train_test_split
    TF_AVAILABLE = True
    print("✅ TensorFlow and scikit-learn successfully imported")
except ImportError as e:
    print(f"⚠️ TensorFlow/scikit-learn not available: {e}")
    TF_AVAILABLE = False

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import project modules
from src.data_processing.data_loader import load_goes_data, preprocess_xrs_data, remove_background
from src.flare_detection.traditional_detection import detect_flare_peaks, define_flare_bounds
from src.ml_models.enhanced_flare_analysis import EnhancedFlareDetector, FlareEnergyAnalyzer, NanoflareDetector
from config.settings import *

class FlareModelTrainer:
    """Comprehensive trainer for solar flare analysis models."""
    
    def __init__(self):
        """Initialize the model trainer."""
        self.models = {}
        self.training_history = {}
        self.evaluation_metrics = {}
        
        # Ensure model directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"🏗️  Model trainer initialized. Models will be saved to: {MODEL_DIR}")
    
    def load_and_prepare_data(self):
        """Load GOES XRS data and prepare it for training."""
        print("\n📊 Loading and preparing GOES XRS data...")
        
        # Find available data files
        data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.nc')]
        
        if not data_files:
            print("❌ No GOES XRS data files found in data directory!")
            print(f"   Please place .nc files in: {DATA_DIR}")
            return None, None
        
        print(f"📂 Found {len(data_files)} data files:")
        for file in data_files:
            print(f"   - {file}")
        
        # Load and combine data from all files
        all_data = []
        
        for file in data_files:
            file_path = os.path.join(DATA_DIR, file)
            try:
                print(f"📈 Loading {file}...")
                
                # Load GOES dataset
                dataset = load_goes_data(file_path)
                
                if dataset is not None:
                    # Preprocess XRS data for both channels
                    for channel in ['A', 'B']:
                        df = preprocess_xrs_data(dataset, channel=channel)
                        if df is not None and len(df) > 100:
                            # Remove background
                            df_clean = remove_background(df, 
                                                       window_size=BACKGROUND_PARAMS['window_size'],
                                                       quantile=BACKGROUND_PARAMS['quantile'])
                            if df_clean is not None:
                                df_clean['channel'] = channel
                                df_clean['file_source'] = file
                                all_data.append(df_clean)
                                print(f"   ✅ Channel {channel}: {len(df_clean)} data points")
                
            except Exception as e:
                print(f"   ⚠️ Error loading {file}: {e}")
                continue
        
        if not all_data:
            print("❌ No usable data found!")
            return None, None
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_index()
        
        print(f"✅ Combined dataset: {len(combined_df)} total data points")
        print(f"   📅 Time range: {combined_df.index.min()} to {combined_df.index.max()}")
        print(f"   🌟 Flux range: {combined_df['flux'].min():.2e} to {combined_df['flux'].max():.2e} W/m²")
        
        return combined_df, data_files
    
    def generate_synthetic_data(self, n_samples=10000, sequence_length=128):
        """Generate synthetic training data for flare decomposition."""
        print(f"\n🧪 Generating {n_samples} synthetic training samples...")
        
        X_synthetic = []
        y_synthetic = []
        
        for i in range(n_samples):
            if i % 1000 == 0:
                print(f"   Progress: {i}/{n_samples} samples")
            
            # Generate synthetic time series with overlapping flares
            time_series, true_components = self._create_synthetic_flare_sequence(sequence_length)
            
            X_synthetic.append(time_series)
            y_synthetic.append(true_components)
        
        X_synthetic = np.array(X_synthetic)
        y_synthetic = np.array(y_synthetic)
        
        print(f"✅ Synthetic data generated:")
        print(f"   📊 Input shape: {X_synthetic.shape}")
        print(f"   🎯 Output shape: {y_synthetic.shape}")
        
        return X_synthetic, y_synthetic
    
    def _create_synthetic_flare_sequence(self, length=128):
        """Create a synthetic time series with known flare components."""
        
        # Initialize background level
        background = 1e-7 + np.random.normal(0, 1e-8, length)
        background = np.maximum(background, 1e-9)  # Ensure positive
        
        # Number of overlapping flares (0-3)
        n_flares = np.random.randint(0, 4)
        
        total_signal = background.copy()
        true_components = np.zeros((length, ML_PARAMS['max_flares']))
        
        for flare_idx in range(n_flares):
            # Random flare parameters
            peak_time = np.random.randint(20, length - 20)
            peak_flux = 10**(np.random.uniform(-6, -4))  # 1e-6 to 1e-4 W/m²
            rise_time = np.random.randint(5, 25)
            decay_time = np.random.randint(10, 50)
            
            # Create flare profile
            flare_profile = self._create_flare_profile(length, peak_time, peak_flux, rise_time, decay_time)
            
            # Add to total signal and store component
            total_signal += flare_profile
            if flare_idx < ML_PARAMS['max_flares']:
                true_components[:, flare_idx] = flare_profile
        
        # Add noise
        noise = np.random.normal(0, np.std(total_signal) * 0.1, length)
        total_signal += noise
        
        # Normalize
        total_signal = np.maximum(total_signal, 1e-9)
        
        return total_signal, true_components
    
    def _create_flare_profile(self, length, peak_time, peak_flux, rise_time, decay_time):
        """Create a realistic flare temporal profile."""
        profile = np.zeros(length)
        
        for i in range(length):
            if i <= peak_time:
                # Rising phase (exponential)
                if i >= peak_time - rise_time:
                    progress = (i - (peak_time - rise_time)) / rise_time
                    profile[i] = peak_flux * (np.exp(progress) - 1) / (np.e - 1)
            else:
                # Decay phase (exponential decay)
                if i <= peak_time + decay_time:
                    progress = (i - peak_time) / decay_time
                    profile[i] = peak_flux * np.exp(-2 * progress)
        
        return profile
    def build_flare_decomposition_model(self, input_shape):
        """Build a neural network for flare decomposition."""
        print("\n🧠 Building flare decomposition model...")
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for model building but not available")
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=input_shape),
            
            # Reshape for CNN
            layers.Reshape((input_shape[0], 1)),
            
            # Convolutional layers for feature extraction
            layers.Conv1D(32, 5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(64, 5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.UpSampling1D(2),
            
            # Output layers for each flare component
            layers.Conv1D(ML_PARAMS['max_flares'], 1, activation='relu', padding='same'),
            
            # Reshape to final output
            layers.Reshape((input_shape[0], ML_PARAMS['max_flares']))
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"✅ Model built with {model.count_params()} parameters")
        model.summary()
        
        return model
    def build_energy_estimation_model(self, input_shape):
        """Build a neural network for flare energy estimation."""
        print("\n⚡ Building energy estimation model...")
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for model building but not available")
        
        model = models.Sequential([
            layers.Input(shape=input_shape),
            
            # Feature extraction
            layers.Conv1D(64, 7, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            
            layers.Conv1D(128, 5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            
            layers.Conv1D(256, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            
            # Dense layers for regression
            layers.Dense(128, activation='relu'),
            layers.Dropout(ML_PARAMS['dropout_rate']),
            layers.Dense(64, activation='relu'),
            layers.Dropout(ML_PARAMS['dropout_rate']),
            layers.Dense(1, activation='linear')  # Energy output
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"✅ Energy model built with {model.count_params()} parameters")
        
        return model
    def train_models(self, real_data=None):
        """Train all neural network models."""
        print("\n🚀 Starting model training pipeline...")
        
        if not TF_AVAILABLE:
            print("❌ TensorFlow is not available. Cannot train neural network models.")
            return None, None
        
        # Generate synthetic training data
        X_synthetic, y_synthetic = self.generate_synthetic_data(
            n_samples=5000, 
            sequence_length=ML_PARAMS['sequence_length']
        )
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_synthetic, y_synthetic, 
            test_size=ML_PARAMS['validation_split'], 
            random_state=42
        )
        
        print(f"📊 Training data split:")
        print(f"   🏋️  Training: {X_train.shape[0]} samples")
        print(f"   ✅ Validation: {X_val.shape[0]} samples")
        
        # 1. Train flare decomposition model
        print("\n1️⃣ Training flare decomposition model...")
        
        decomp_model = self.build_flare_decomposition_model((ML_PARAMS['sequence_length'],))
        
        # Callbacks
        callbacks_decomp = [
            callbacks.EarlyStopping(
                patience=ML_PARAMS['early_stopping_patience'], 
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
            callbacks.ModelCheckpoint(
                os.path.join(MODEL_DIR, 'flare_decomposition_model.h5'),
                save_best_only=True
            )
        ]
        
        # Train
        history_decomp = decomp_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=ML_PARAMS['epochs'],
            batch_size=ML_PARAMS['batch_size'],
            callbacks=callbacks_decomp,
            verbose=1
        )
        
        self.models['decomposition'] = decomp_model
        self.training_history['decomposition'] = history_decomp
        
        # 2. Train energy estimation model
        print("\n2️⃣ Training energy estimation model...")
        
        # Generate energy labels for synthetic data
        y_energy_train = self._calculate_synthetic_energies(y_train)
        y_energy_val = self._calculate_synthetic_energies(y_val)
        
        energy_model = self.build_energy_estimation_model((ML_PARAMS['sequence_length'],))
        
        callbacks_energy = [
            callbacks.EarlyStopping(
                patience=ML_PARAMS['early_stopping_patience'], 
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
            callbacks.ModelCheckpoint(
                os.path.join(MODEL_DIR, 'energy_estimation_model.h5'),
                save_best_only=True
            )
        ]
        
        # Train
        history_energy = energy_model.fit(
            X_train, y_energy_train,
            validation_data=(X_val, y_energy_val),
            epochs=ML_PARAMS['epochs']//2,  # Fewer epochs for energy model
            batch_size=ML_PARAMS['batch_size'],
            callbacks=callbacks_energy,
            verbose=1
        )
        
        self.models['energy'] = energy_model
        self.training_history['energy'] = history_energy
        
        print("✅ Model training completed!")
        
        return self.models, self.training_history
    
    def _calculate_synthetic_energies(self, y_components):
        """Calculate synthetic energy labels from flare components."""
        energies = []
        
        for i in range(y_components.shape[0]):
            total_energy = 0
            for j in range(y_components.shape[2]):  # Each flare component
                component = y_components[i, :, j]
                # Simple energy calculation (integral of flux over time)
                energy = np.trapz(component) * 60  # Assuming 1-minute time steps
                total_energy += energy
            
            energies.append(total_energy)
        
        return np.array(energies)
    
    def evaluate_models(self, real_data=None):
        """Evaluate trained models and generate performance metrics."""
        print("\n📊 Evaluating model performance...")
        
        if not self.models:
            print("❌ No trained models found! Please train models first.")
            return
        
        # Generate test data
        X_test, y_test = self.generate_synthetic_data(n_samples=1000, 
                                                     sequence_length=ML_PARAMS['sequence_length'])
        
        # Evaluate decomposition model
        if 'decomposition' in self.models:
            print("\n🔍 Evaluating flare decomposition model...")
            decomp_model = self.models['decomposition']
            
            # Predictions
            y_pred = decomp_model.predict(X_test, verbose=0)
            
            # Calculate metrics
            mse = np.mean((y_test - y_pred)**2)
            mae = np.mean(np.abs(y_test - y_pred))
            r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
            
            self.evaluation_metrics['decomposition'] = {
                'mse': mse,
                'mae': mae,
                'r2': r2
            }
            
            print(f"   📈 MSE: {mse:.2e}")
            print(f"   📊 MAE: {mae:.2e}")
            print(f"   🎯 R²: {r2:.4f}")
        
        # Evaluate energy model
        if 'energy' in self.models:
            print("\n⚡ Evaluating energy estimation model...")
            energy_model = self.models['energy']
            
            # Generate energy labels
            y_energy_test = self._calculate_synthetic_energies(y_test)
            
            # Predictions
            y_energy_pred = energy_model.predict(X_test, verbose=0).flatten()
            
            # Calculate metrics
            mse_energy = np.mean((y_energy_test - y_energy_pred)**2)
            mae_energy = np.mean(np.abs(y_energy_test - y_energy_pred))
            r2_energy = 1 - np.sum((y_energy_test - y_energy_pred)**2) / np.sum((y_energy_test - np.mean(y_energy_test))**2)
            
            self.evaluation_metrics['energy'] = {
                'mse': mse_energy,
                'mae': mae_energy,
                'r2': r2_energy
            }
            
            print(f"   📈 MSE: {mse_energy:.2e}")
            print(f"   📊 MAE: {mae_energy:.2e}")
            print(f"   🎯 R²: {r2_energy:.4f}")
        
        # Save evaluation results
        self._save_evaluation_results()
        
        return self.evaluation_metrics
    
    def _save_evaluation_results(self):
        """Save evaluation results to file."""
        results_file = os.path.join(OUTPUT_DIR, 'model_evaluation_results.txt')
        
        with open(results_file, 'w') as f:
            f.write("Solar Flare Analysis - Model Evaluation Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for model_name, metrics in self.evaluation_metrics.items():
                f.write(f"{model_name.upper()} MODEL:\n")
                f.write("-" * 20 + "\n")
                for metric_name, value in metrics.items():
                    f.write(f"{metric_name.upper()}: {value:.4e}\n")
                f.write("\n")
        
        print(f"💾 Evaluation results saved to: {results_file}")
    
    def save_models(self):
        """Save all trained models."""
        print("\n💾 Saving trained models...")
        
        for model_name, model in self.models.items():
            model_path = os.path.join(MODEL_DIR, f'{model_name}_model.h5')
            model.save(model_path)
            print(f"   ✅ {model_name} model saved to: {model_path}")
        
        # Save training history
        history_file = os.path.join(OUTPUT_DIR, 'training_history.npz')
        history_data = {}
        
        for model_name, history in self.training_history.items():
            for key, values in history.history.items():
                history_data[f'{model_name}_{key}'] = values
        
        np.savez(history_file, **history_data)
        print(f"   📊 Training history saved to: {history_file}")
    
    def plot_training_results(self):
        """Plot training history and model performance."""
        print("\n📈 Generating training plots...")
        
        # Create plots directory
        plots_dir = os.path.join(OUTPUT_DIR, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot training history for each model
        for model_name, history in self.training_history.items():
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Loss plot
            axes[0].plot(history.history['loss'], label='Training Loss')
            axes[0].plot(history.history['val_loss'], label='Validation Loss')
            axes[0].set_title(f'{model_name.title()} Model - Training Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True)
            
            # MAE plot
            axes[1].plot(history.history['mae'], label='Training MAE')
            axes[1].plot(history.history['val_mae'], label='Validation MAE')
            axes[1].set_title(f'{model_name.title()} Model - Mean Absolute Error')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE')
            axes[1].legend()
            axes[1].grid(True)
            
            plt.tight_layout()
            
            plot_path = os.path.join(plots_dir, f'{model_name}_training_history.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   📊 {model_name} training plot saved to: {plot_path}")
    
    def apply_to_real_data(self, real_data):
        """Apply trained models to real GOES XRS data."""
        print("\n🌞 Applying models to real GOES XRS data...")
        
        if not self.models:
            print("❌ No trained models found! Please train models first.")
            return None
        
        if real_data is None or len(real_data) == 0:
            print("❌ No real data provided!")
            return None
        
        # Prepare real data for prediction
        sequence_length = ML_PARAMS['sequence_length']
        flux_data = real_data['flux'].values
        
        # Create sequences
        sequences = []
        indices = []
        
        for i in range(len(flux_data) - sequence_length + 1):
            sequences.append(flux_data[i:i + sequence_length])
            indices.append(i + sequence_length - 1)
        
        X_real = np.array(sequences)
        
        print(f"   📊 Created {len(X_real)} sequences from real data")
        
        # Apply decomposition model
        results = {}
        
        if 'decomposition' in self.models:
            print("   🔍 Applying flare decomposition model...")
            decomp_predictions = self.models['decomposition'].predict(X_real, verbose=0)
            results['decomposition'] = decomp_predictions
        
        if 'energy' in self.models:
            print("   ⚡ Applying energy estimation model...")
            energy_predictions = self.models['energy'].predict(X_real, verbose=0)
            results['energy'] = energy_predictions
        
        # Save results
        results_file = os.path.join(OUTPUT_DIR, 'real_data_predictions.npz')
        np.savez(results_file, 
                indices=indices,
                **results)
        
        print(f"   💾 Predictions saved to: {results_file}")
        
        return results


def main():
    """Main training pipeline."""
    print("🌞 Solar Flare Analysis - Neural Network Training Pipeline")
    print("=" * 70)
    
    # Initialize trainer
    trainer = FlareModelTrainer()
    
    # Load real data
    real_data, data_files = trainer.load_and_prepare_data()
    
    # Train models
    models, history = trainer.train_models(real_data)
    
    # Evaluate models
    metrics = trainer.evaluate_models(real_data)
    
    # Save models
    trainer.save_models()
    
    # Generate plots
    trainer.plot_training_results()
    
    # Apply to real data if available
    if real_data is not None:
        results = trainer.apply_to_real_data(real_data)
    
    print("\n🎉 Training pipeline completed successfully!")
    print(f"📂 Models saved to: {MODEL_DIR}")
    print(f"📊 Results saved to: {OUTPUT_DIR}")
    
    # List saved models
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.h5')]
    if model_files:
        print(f"\n💾 Saved models:")
        for model_file in model_files:
            print(f"   - {model_file}")
    
    return trainer


if __name__ == "__main__":
    trainer = main()

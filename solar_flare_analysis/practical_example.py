#!/usr/bin/env python3
"""
Practical Example: Using .h5 Models with Real GOES Data

This script demonstrates how to:
1. Load GOES XRS data from netCDF files
2. Preprocess the data for model input
3. Use trained models for solar flare prediction
4. Visualize results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

# Try to import netCDF4 - if not available, provide instructions
try:
    import netCDF4 as nc
    HAS_NETCDF = True
except ImportError:
    HAS_NETCDF = False
    print("⚠️ netCDF4 not found. Install with: pip install netcdf4")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GOESDataLoader:
    """Load and preprocess GOES XRS data for model inference."""
    
    def __init__(self):
        self.data = None
        self.processed_features = None
    
    def load_netcdf_file(self, file_path: str) -> pd.DataFrame:
        """Load GOES data from netCDF file."""
        if not HAS_NETCDF:
            raise ImportError("netCDF4 is required to read GOES data files")
        
        try:
            with nc.Dataset(file_path, 'r') as dataset:
                print(f"📡 Loading GOES data from: {Path(file_path).name}")
                
                # Print available variables
                print("Available variables:", list(dataset.variables.keys()))
                
                # Try common GOES variable names
                time_var = None
                xrs_a_var = None
                xrs_b_var = None
                
                # Look for time variables
                for var in ['time', 'time_tag', 't']:
                    if var in dataset.variables:
                        time_var = var
                        break
                
                # Look for XRS-A variables (0.05-0.4 nm)
                for var in ['xrsa_flux', 'XRS_A', 'A_FLUX', 'xrs_a']:
                    if var in dataset.variables:
                        xrs_a_var = var
                        break
                
                # Look for XRS-B variables (0.1-0.8 nm)
                for var in ['xrsb_flux', 'XRS_B', 'B_FLUX', 'xrs_b']:
                    if var in dataset.variables:
                        xrs_b_var = var
                        break
                
                if not all([time_var, xrs_a_var, xrs_b_var]):
                    print("❌ Could not find required variables")
                    print(f"Time variable: {time_var}")
                    print(f"XRS-A variable: {xrs_a_var}")
                    print(f"XRS-B variable: {xrs_b_var}")
                    return None
                
                # Extract data
                times = dataset.variables[time_var][:]
                xrs_a = dataset.variables[xrs_a_var][:]
                xrs_b = dataset.variables[xrs_b_var][:]
                
                # Handle time conversion
                try:
                    if hasattr(dataset.variables[time_var], 'units'):
                        time_units = dataset.variables[time_var].units
                        if 'seconds since' in time_units:
                            base_time = pd.to_datetime(time_units.split('since ')[1])
                            timestamps = pd.to_datetime(times, unit='s', origin=base_time)
                        else:
                            # Fallback: assume GOES epoch (2000-01-01 12:00:00)
                            base_time = pd.to_datetime('2000-01-01 12:00:00')
                            timestamps = pd.to_datetime(times, unit='s', origin=base_time)
                    else:
                        # Fallback time conversion
                        base_time = pd.to_datetime('2000-01-01 12:00:00')
                        timestamps = pd.to_datetime(times, unit='s', origin=base_time)
                except:
                    # If time conversion fails, create a simple time series
                    timestamps = pd.date_range(start='2023-01-01', periods=len(times), freq='2S')
                
                # Create DataFrame
                data = pd.DataFrame({
                    'timestamp': timestamps,
                    'xrs_a': xrs_a,
                    'xrs_b': xrs_b
                })
                
                # Add derived features
                data['ratio'] = data['xrs_a'] / (data['xrs_b'] + 1e-12)
                data['xrs_a_log'] = np.log10(data['xrs_a'] + 1e-12)
                data['xrs_b_log'] = np.log10(data['xrs_b'] + 1e-12)
                
                # Clean data
                initial_count = len(data)
                data = data.dropna()
                data = data[data['xrs_a'] > 0]
                data = data[data['xrs_b'] > 0]
                
                print(f"✅ Loaded {len(data)} valid data points (from {initial_count} total)")
                print(f"Time range: {data['timestamp'].min()} to {data['timestamp'].max()}")
                print(f"XRS-A range: {data['xrs_a'].min():.2e} to {data['xrs_a'].max():.2e}")
                print(f"XRS-B range: {data['xrs_b'].min():.2e} to {data['xrs_b'].max():.2e}")
                
                self.data = data
                return data
                
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return None
    
    def load_csv_file(self, file_path: str) -> pd.DataFrame:
        """Load processed GOES data from CSV file."""
        try:
            data = pd.read_csv(file_path)
            
            # Ensure timestamp column
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Add derived features if not present
            if 'ratio' not in data.columns and 'xrs_a' in data.columns and 'xrs_b' in data.columns:
                data['ratio'] = data['xrs_a'] / (data['xrs_b'] + 1e-12)
            
            if 'xrs_a_log' not in data.columns and 'xrs_a' in data.columns:
                data['xrs_a_log'] = np.log10(data['xrs_a'] + 1e-12)
            
            if 'xrs_b_log' not in data.columns and 'xrs_b' in data.columns:
                data['xrs_b_log'] = np.log10(data['xrs_b'] + 1e-12)
            
            print(f"✅ Loaded {len(data)} data points from CSV")
            self.data = data
            return data
            
        except Exception as e:
            logger.error(f"❌ Error loading CSV: {e}")
            return None
    
    def create_sliding_windows(self, window_size: int = 60, step_size: int = 1) -> np.ndarray:
        """Create sliding windows for time series prediction."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_netcdf_file() or load_csv_file() first.")
        
        # Select features for model input
        feature_columns = ['xrs_a_log', 'xrs_b_log', 'ratio']
        available_features = [col for col in feature_columns if col in self.data.columns]
        
        if not available_features:
            raise ValueError(f"No suitable features found. Available columns: {self.data.columns.tolist()}")
        
        print(f"Using features: {available_features}")
        
        # Extract feature data
        feature_data = self.data[available_features].values
        
        # Create sliding windows
        windows = []
        for i in range(0, len(feature_data) - window_size + 1, step_size):
            window = feature_data[i:i + window_size]
            windows.append(window.flatten())  # Flatten for model input
        
        windows = np.array(windows)
        print(f"Created {len(windows)} windows of size {window_size}")
        print(f"Window shape: {windows.shape}")
        
        self.processed_features = windows
        return windows
    
    def get_latest_window(self, window_size: int = 60) -> np.ndarray:
        """Get the most recent data window for real-time prediction."""
        if self.data is None:
            raise ValueError("No data loaded")
        
        feature_columns = ['xrs_a_log', 'xrs_b_log', 'ratio']
        available_features = [col for col in feature_columns if col in self.data.columns]
        
        # Get the last window_size points
        recent_data = self.data[available_features].tail(window_size).values
        
        if len(recent_data) < window_size:
            # Pad if not enough data
            padding = np.zeros((window_size - len(recent_data), recent_data.shape[1]))
            recent_data = np.vstack([padding, recent_data])
        
        return recent_data.flatten().reshape(1, -1)  # Shape for model input

class FlarePredictor:
    """Use trained models to predict solar flares."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load available trained models."""
        model_files = [
            ('binary_classifier', 'binary_flare_classifier.h5'),
            ('multiclass_classifier', 'multiclass_flare_classifier.h5'),
            ('energy_estimator', 'energy_regression_model.h5'),
            ('cnn_detector', 'cnn_flare_detector.h5'),
            ('minimal_detector', 'minimal_flare_model.h5')
        ]
        
        for model_name, filename in model_files:
            model_path = self.models_dir / filename
            if model_path.exists():
                try:
                    self.models[model_name] = keras.models.load_model(str(model_path))
                    print(f"✅ Loaded {model_name}")
                except Exception as e:
                    print(f"❌ Failed to load {model_name}: {e}")
    
    def predict_single_window(self, window_data: np.ndarray) -> dict:
        """Predict on a single time window."""
        results = {}
        
        for model_name, model in self.models.items():
            try:
                # Adjust input shape for different model types
                if model_name == 'cnn_detector':
                    # CNN needs 3D input
                    model_input = window_data.reshape(window_data.shape[0], -1, 1)
                else:
                    model_input = window_data
                
                prediction = model.predict(model_input, verbose=0)
                
                # Interpret results based on model type
                if model_name in ['binary_classifier', 'minimal_detector']:
                    prob = float(prediction[0, 0])
                    results[model_name] = {
                        'probability': prob,
                        'classification': 'flare' if prob > 0.5 else 'no_flare',
                        'confidence': abs(prob - 0.5) * 2
                    }
                
                elif model_name == 'multiclass_classifier':
                    classes = ['no_flare', 'C_class', 'M_class', 'X_class']
                    probs = prediction[0]
                    class_idx = np.argmax(probs)
                    results[model_name] = {
                        'predicted_class': classes[class_idx],
                        'probabilities': {classes[i]: float(probs[i]) for i in range(len(classes))},
                        'confidence': float(np.max(probs))
                    }
                
                elif model_name == 'energy_estimator':
                    energy = float(prediction[0, 0])
                    results[model_name] = {
                        'estimated_energy': energy,
                        'energy_log10': np.log10(max(energy, 1e-10))
                    }
                
                elif model_name == 'cnn_detector':
                    score = float(prediction[0, 0])
                    results[model_name] = {
                        'detection_score': score,
                        'detected': score > 0.5
                    }
                    
            except Exception as e:
                results[model_name] = {'error': str(e)}
        
        return results
    
    def predict_batch(self, window_data: np.ndarray) -> list:
        """Predict on multiple time windows."""
        batch_results = []
        
        for i, window in enumerate(window_data):
            window_reshaped = window.reshape(1, -1)
            result = self.predict_single_window(window_reshaped)
            result['window_index'] = i
            batch_results.append(result)
        
        return batch_results

def plot_data_and_predictions(data_loader: GOESDataLoader, predictions: list, save_path: str = None):
    """Plot GOES data with prediction results."""
    
    if data_loader.data is None:
        print("❌ No data to plot")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot XRS data
    axes[0].plot(data_loader.data['timestamp'], data_loader.data['xrs_a'], 
                label='XRS-A (0.05-0.4 nm)', color='red', alpha=0.7)
    axes[0].plot(data_loader.data['timestamp'], data_loader.data['xrs_b'], 
                label='XRS-B (0.1-0.8 nm)', color='blue', alpha=0.7)
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Flux (W/m²)')
    axes[0].set_title('GOES XRS Data')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot ratio
    axes[1].plot(data_loader.data['timestamp'], data_loader.data['ratio'], 
                color='green', alpha=0.7)
    axes[1].set_ylabel('XRS-A/XRS-B Ratio')
    axes[1].set_title('XRS Ratio')
    axes[1].grid(True, alpha=0.3)
    
    # Plot predictions if available
    if predictions:
        # Extract binary classification probabilities
        binary_probs = []
        timestamps = []
        
        for i, pred in enumerate(predictions):
            if 'binary_classifier' in pred and 'error' not in pred['binary_classifier']:
                prob = pred['binary_classifier']['probability']
                binary_probs.append(prob)
                # Approximate timestamp for this prediction
                if i < len(data_loader.data):
                    timestamps.append(data_loader.data['timestamp'].iloc[i])
        
        if binary_probs:
            axes[2].plot(timestamps, binary_probs, color='purple', linewidth=2)
            axes[2].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Threshold')
            axes[2].set_ylabel('Flare Probability')
            axes[2].set_title('Binary Flare Predictions')
            axes[2].set_ylim(0, 1)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved to: {save_path}")
    else:
        plt.show()

def main():
    """Main function demonstrating the complete workflow."""
    
    print("🌟 Practical GOES Data + Model Usage Example")
    print("=" * 50)
    
    # Initialize components
    data_loader = GOESDataLoader()
    predictor = FlarePredictor()
    
    if not predictor.models:
        print("❌ No models loaded. Check your models directory.")
        return
    
    # Look for data files
    data_dir = Path("data_cache")
    enhanced_dir = Path("enhanced_output")
    
    data_file = None
    
    # Try to find a data file
    for data_path in [data_dir, enhanced_dir, Path(".")]:
        if data_path.exists():
            # Look for netCDF files
            nc_files = list(data_path.glob("*.nc"))
            if nc_files:
                data_file = nc_files[0]
                break
            
            # Look for CSV files
            csv_files = list(data_path.glob("*.csv"))
            if csv_files:
                data_file = csv_files[0]
                break
    
    if not data_file:
        print("❌ No data files found. Creating synthetic data for demonstration...")
        
        # Create synthetic data
        timestamps = pd.date_range('2023-01-01', periods=1000, freq='2S')
        # Simulate quiet background with some flare events
        xrs_a = np.random.lognormal(-20, 1, 1000)
        xrs_b = np.random.lognormal(-19, 1, 1000)
        
        # Add some "flare" events
        flare_times = [200, 400, 700]
        for flare_time in flare_times:
            if flare_time < len(xrs_a):
                # Simulate flare enhancement
                xrs_a[flare_time:flare_time+50] *= np.exp(np.random.random(50) * 5)
                xrs_b[flare_time:flare_time+50] *= np.exp(np.random.random(50) * 4)
        
        synthetic_data = pd.DataFrame({
            'timestamp': timestamps,
            'xrs_a': xrs_a,
            'xrs_b': xrs_b
        })
        
        # Add derived features
        synthetic_data['ratio'] = synthetic_data['xrs_a'] / (synthetic_data['xrs_b'] + 1e-12)
        synthetic_data['xrs_a_log'] = np.log10(synthetic_data['xrs_a'] + 1e-12)
        synthetic_data['xrs_b_log'] = np.log10(synthetic_data['xrs_b'] + 1e-12)
        
        data_loader.data = synthetic_data
        print("✅ Created synthetic GOES-like data")
        
    else:
        print(f"📁 Found data file: {data_file}")
        
        # Load the data file
        if data_file.suffix == '.nc':
            data_loader.load_netcdf_file(str(data_file))
        elif data_file.suffix == '.csv':
            data_loader.load_csv_file(str(data_file))
    
    if data_loader.data is None:
        print("❌ Failed to load data")
        return
    
    # Create sliding windows
    print("\n🔄 Processing data for model input...")
    windows = data_loader.create_sliding_windows(window_size=60, step_size=10)
    
    # Make predictions on a subset of windows (for demo)
    n_predict = min(10, len(windows))
    print(f"\n🔮 Making predictions on {n_predict} windows...")
    
    predictions = predictor.predict_batch(windows[:n_predict])
    
    # Display results
    print("\n📊 Prediction Results:")
    print("=" * 30)
    
    for i, pred in enumerate(predictions[:5]):  # Show first 5
        print(f"\nWindow {i+1}:")
        for model_name, result in pred.items():
            if model_name == 'window_index':
                continue
            
            if 'error' in result:
                print(f"  ❌ {model_name}: {result['error']}")
            else:
                if model_name == 'binary_classifier':
                    print(f"  🔸 Binary: {result['classification']} ({result['probability']:.3f})")
                elif model_name == 'multiclass_classifier':
                    print(f"  🔸 Multi: {result['predicted_class']} ({result['confidence']:.3f})")
                elif model_name == 'energy_estimator':
                    print(f"  🔸 Energy: {result['estimated_energy']:.2e} W/m²")
    
    # Create visualization
    print("\n📈 Creating visualization...")
    plot_data_and_predictions(data_loader, predictions, "prediction_results.png")
    
    print("\n✅ Complete workflow demonstrated!")
    print("\nFiles created:")
    print("- prediction_results.png: Visualization of data and predictions")
    print("\nNext Steps:")
    print("1. Adapt data loading for your specific GOES file format")
    print("2. Tune window size and features based on your models")
    print("3. Implement real-time prediction pipeline")
    print("4. Add threshold tuning and alert system")

if __name__ == "__main__":
    main()

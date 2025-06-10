#!/usr/bin/env python3
"""
Comprehensive Guide: Using Trained .h5 Model Files in Solar Flare Analysis

This guide demonstrates how to load and use the trained models in your 
/solar_flare_analysis/models/ folder for solar flare prediction and analysis.

Available Models:
- binary_flare_classifier.h5: Binary classification (flare/no-flare)
- multiclass_flare_classifier.h5: Multi-class classification (C, M, X class flares)
- energy_regression_model.h5: Energy estimation regression
- cnn_flare_detector.h5: CNN-based flare detection
- minimal_flare_model.h5: Minimal/lightweight flare detection
- test_model.h5: Test/experimental model

Author: Solar Flare Analysis Team
Date: June 2025
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import netCDF4 as nc
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SolarFlareModelManager:
    """
    Manages loading and inference of trained solar flare models.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.model_info = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all available trained models from the models directory."""
        model_files = {
            'binary_classifier': 'binary_flare_classifier.h5',
            'multiclass_classifier': 'multiclass_flare_classifier.h5',
            'energy_estimator': 'energy_regression_model.h5',
            'cnn_detector': 'cnn_flare_detector.h5',
            'minimal_detector': 'minimal_flare_model.h5',
            'test_model': 'test_model.h5'
        }
        
        for model_name, filename in model_files.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                try:
                    model = keras.models.load_model(str(model_path))
                    self.models[model_name] = model
                    
                    # Store model information
                    self.model_info[model_name] = {
                        'filename': filename,
                        'input_shape': model.input_shape,
                        'output_shape': model.output_shape,
                        'parameters': model.count_params(),
                        'layers': len(model.layers),
                        'architecture': [layer.__class__.__name__ for layer in model.layers]
                    }
                    logger.info(f"✅ Loaded model: {model_name}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load model {model_name}: {e}")
            else:
                logger.warning(f"⚠️ Model file not found: {model_path}")
        
        logger.info(f"📊 Successfully loaded {len(self.models)} models")
    
    def get_model_summary(self, model_name: str = None):
        """Get summary information about loaded models."""
        if model_name:
            if model_name in self.models:
                model = self.models[model_name]
                print(f"\n🔍 Model Summary: {model_name}")
                print("=" * 50)
                model.summary()
                print(f"\nModel Info: {self.model_info[model_name]}")
            else:
                print(f"❌ Model '{model_name}' not found")
        else:
            print("\n📋 All Loaded Models:")
            print("=" * 50)
            for name, info in self.model_info.items():
                print(f"🔸 {name}:")
                print(f"   - File: {info['filename']}")
                print(f"   - Input Shape: {info['input_shape']}")
                print(f"   - Output Shape: {info['output_shape']}")
                print(f"   - Parameters: {info['parameters']:,}")
                print(f"   - Layers: {info['layers']}")
                print()
    
    def predict_flare_binary(self, data: np.ndarray) -> Dict:
        """Binary flare prediction (flare/no-flare)."""
        if 'binary_classifier' not in self.models:
            return {'error': 'Binary classifier model not loaded'}
        
        try:
            # Ensure proper data shape
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            
            prediction = self.models['binary_classifier'].predict(data, verbose=0)
            probability = float(prediction[0, 0])
            
            return {
                'model': 'binary_classifier',
                'probability': probability,
                'classification': 'flare' if probability > 0.5 else 'no_flare',
                'confidence': float(abs(probability - 0.5) * 2)
            }
        except Exception as e:
            return {'error': f'Binary prediction failed: {e}'}
    
    def predict_flare_multiclass(self, data: np.ndarray) -> Dict:
        """Multi-class flare prediction (C, M, X class)."""
        if 'multiclass_classifier' not in self.models:
            return {'error': 'Multiclass classifier model not loaded'}
        
        try:
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            
            prediction = self.models['multiclass_classifier'].predict(data, verbose=0)
            class_names = ['no_flare', 'C_class', 'M_class', 'X_class']
            class_idx = np.argmax(prediction[0])
            
            return {
                'model': 'multiclass_classifier',
                'predicted_class': class_names[class_idx],
                'probabilities': {
                    class_names[i]: float(prediction[0, i]) 
                    for i in range(len(class_names))
                },
                'confidence': float(np.max(prediction[0]))
            }
        except Exception as e:
            return {'error': f'Multiclass prediction failed: {e}'}
    
    def predict_flare_energy(self, data: np.ndarray) -> Dict:
        """Predict flare energy estimation."""
        if 'energy_estimator' not in self.models:
            return {'error': 'Energy estimator model not loaded'}
        
        try:
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            
            prediction = self.models['energy_estimator'].predict(data, verbose=0)
            energy = float(prediction[0, 0])
            
            return {
                'model': 'energy_estimator',
                'estimated_energy': energy,
                'energy_log10': float(np.log10(max(energy, 1e-10))),
                'energy_watts_per_m2': energy
            }
        except Exception as e:
            return {'error': f'Energy prediction failed: {e}'}
    
    def predict_cnn_detection(self, data: np.ndarray) -> Dict:
        """CNN-based flare detection."""
        if 'cnn_detector' not in self.models:
            return {'error': 'CNN detector model not loaded'}
        
        try:
            # Reshape for CNN (add channel dimension)
            if len(data.shape) == 2:
                data = data.reshape(data.shape[0], data.shape[1], 1)
            elif len(data.shape) == 1:
                data = data.reshape(1, -1, 1)
            
            prediction = self.models['cnn_detector'].predict(data, verbose=0)
            detection_score = float(prediction[0, 0])
            
            return {
                'model': 'cnn_detector',
                'detection_score': detection_score,
                'detected': bool(detection_score > 0.5),
                'confidence': float(abs(detection_score - 0.5) * 2)
            }
        except Exception as e:
            return {'error': f'CNN detection failed: {e}'}
    
    def predict_all_models(self, data: np.ndarray) -> Dict:
        """Run predictions using all available models."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'input_shape': data.shape,
            'predictions': {}
        }
        
        # Binary classification
        binary_result = self.predict_flare_binary(data)
        results['predictions']['binary'] = binary_result
        
        # Multiclass classification
        multiclass_result = self.predict_flare_multiclass(data)
        results['predictions']['multiclass'] = multiclass_result
        
        # Energy estimation
        energy_result = self.predict_flare_energy(data)
        results['predictions']['energy'] = energy_result
        
        # CNN detection
        cnn_result = self.predict_cnn_detection(data)
        results['predictions']['cnn'] = cnn_result
        
        # Minimal detection
        if 'minimal_detector' in self.models:
            try:
                if len(data.shape) == 1:
                    data_reshaped = data.reshape(1, -1)
                else:
                    data_reshaped = data
                minimal_pred = self.models['minimal_detector'].predict(data_reshaped, verbose=0)
                results['predictions']['minimal'] = {
                    'model': 'minimal_detector',
                    'detection_score': float(minimal_pred[0, 0]),
                    'detected': bool(minimal_pred[0, 0] > 0.5)
                }
            except Exception as e:
                results['predictions']['minimal'] = {'error': str(e)}
        
        return results

class DataProcessor:
    """
    Handles loading and preprocessing of GOES XRS data for model inference.
    """
    
    def __init__(self):
        self.current_data = None
        self.processed_data = None
    
    def load_goes_data(self, file_path: str) -> pd.DataFrame:
        """Load GOES XRS data from netCDF file."""
        try:
            with nc.Dataset(file_path, 'r') as dataset:
                # Extract variables
                times = dataset.variables['time'][:]
                xrs_a = dataset.variables['xrsa_flux'][:]  # 0.05-0.4 nm
                xrs_b = dataset.variables['xrsb_flux'][:]  # 0.1-0.8 nm
                
                # Convert time to datetime
                base_time = pd.to_datetime('2000-01-01 12:00:00')  # GOES epoch
                timestamps = pd.to_datetime(times, unit='s', origin=base_time)
                
                # Create DataFrame
                data = pd.DataFrame({
                    'timestamp': timestamps,
                    'xrs_a': xrs_a,
                    'xrs_b': xrs_b,
                    'ratio': xrs_a / (xrs_b + 1e-12)
                })
                
                # Clean data
                data = data.dropna()
                data = data[data['xrs_a'] > 0]
                data = data[data['xrs_b'] > 0]
                
                self.current_data = data
                logger.info(f"✅ Loaded {len(data)} data points from {file_path}")
                return data
                
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return None
    
    def preprocess_for_models(self, data: pd.DataFrame, 
                            window_size: int = 60,
                            features: List[str] = None) -> np.ndarray:
        """Preprocess data for model inference."""
        if features is None:
            features = ['xrs_a', 'xrs_b', 'ratio']
        
        try:
            # Extract features
            feature_data = data[features].values
            
            # Normalize data (log scale for XRS values)
            if 'xrs_a' in features:
                xrs_a_idx = features.index('xrs_a')
                feature_data[:, xrs_a_idx] = np.log10(feature_data[:, xrs_a_idx] + 1e-12)
            
            if 'xrs_b' in features:
                xrs_b_idx = features.index('xrs_b')
                feature_data[:, xrs_b_idx] = np.log10(feature_data[:, xrs_b_idx] + 1e-12)
            
            # Create sliding windows
            if len(feature_data) >= window_size:
                # Take the last window for prediction
                processed_data = feature_data[-window_size:]
                processed_data = processed_data.flatten()  # Flatten for model input
            else:
                # Pad if not enough data
                processed_data = np.pad(feature_data.flatten(), 
                                      (0, window_size * len(features) - len(feature_data.flatten())), 
                                      mode='constant', constant_values=0)
            
            self.processed_data = processed_data
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing data: {e}")
            return None

def demonstrate_model_usage():
    """Demonstrate how to use the trained models."""
    print("🚀 Solar Flare Model Usage Demonstration")
    print("=" * 50)
    
    # Initialize model manager
    model_manager = SolarFlareModelManager()
    
    # Show loaded models
    model_manager.get_model_summary()
    
    # Generate sample data for demonstration
    print("\n🔬 Generating Sample Data for Demo...")
    
    # Create synthetic GOES-like data
    sample_data = np.random.lognormal(mean=-8, sigma=1, size=(100, 3))  # Simulated XRS data
    sample_data = sample_data.flatten()  # Flatten for model input
    
    print(f"Sample data shape: {sample_data.shape}")
    
    # Run predictions
    print("\n🔮 Running Predictions...")
    results = model_manager.predict_all_models(sample_data)
    
    # Display results
    print("\n📊 Prediction Results:")
    print("=" * 30)
    
    for model_type, result in results['predictions'].items():
        print(f"\n🔸 {model_type.upper()} MODEL:")
        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            for key, value in result.items():
                if key != 'model':
                    print(f"   - {key}: {value}")
    
    return results

def batch_process_data(data_dir: str, model_manager: SolarFlareModelManager):
    """Process multiple data files in batch."""
    print(f"\n🔄 Batch Processing Data from {data_dir}")
    
    data_processor = DataProcessor()
    results = []
    
    # Find all netCDF files
    data_path = Path(data_dir)
    nc_files = list(data_path.glob("*.nc"))
    
    print(f"Found {len(nc_files)} netCDF files")
    
    for file_path in nc_files[:5]:  # Process first 5 files as example
        print(f"\n📁 Processing: {file_path.name}")
        
        # Load data
        data = data_processor.load_goes_data(str(file_path))
        if data is None:
            continue
        
        # Preprocess
        processed_data = data_processor.preprocess_for_models(data)
        if processed_data is None:
            continue
        
        # Run predictions
        file_results = model_manager.predict_all_models(processed_data)
        file_results['filename'] = file_path.name
        results.append(file_results)
        
        # Quick summary
        predictions = file_results['predictions']
        if 'binary' in predictions and 'error' not in predictions['binary']:
            binary_result = predictions['binary']['classification']
            print(f"   🎯 Binary Classification: {binary_result}")
        
        if 'multiclass' in predictions and 'error' not in predictions['multiclass']:
            multi_result = predictions['multiclass']['predicted_class']
            print(f"   🎯 Multiclass Classification: {multi_result}")
    
    return results

def save_predictions_to_csv(results: List[Dict], output_file: str):
    """Save prediction results to CSV file."""
    rows = []
    
    for result in results:
        filename = result.get('filename', 'unknown')
        timestamp = result.get('timestamp', '')
        
        row = {
            'filename': filename,
            'timestamp': timestamp,
            'input_shape': str(result.get('input_shape', ''))
        }
        
        # Extract predictions
        predictions = result.get('predictions', {})
        
        for model_type, pred_result in predictions.items():
            if 'error' not in pred_result:
                for key, value in pred_result.items():
                    if key != 'model':
                        row[f'{model_type}_{key}'] = value
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"💾 Results saved to: {output_file}")

if __name__ == "__main__":
    # Run demonstration
    demo_results = demonstrate_model_usage()
    
    # Example of batch processing (uncomment to use)
    # model_manager = SolarFlareModelManager()
    # batch_results = batch_process_data("data_cache", model_manager)
    # save_predictions_to_csv(batch_results, "flare_predictions.csv")
    
    print("\n✅ Model usage demonstration completed!")
    print("\nNext steps:")
    print("1. Load your GOES data files")
    print("2. Preprocess data using DataProcessor")
    print("3. Use SolarFlareModelManager for predictions")
    print("4. Analyze and visualize results")

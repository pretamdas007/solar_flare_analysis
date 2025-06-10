#!/usr/bin/env python3
"""
Quick Start: Using Trained .h5 Models for Solar Flare Prediction

This script shows the basic steps to load and use your trained models.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from pathlib import Path

def basic_model_usage():
    """Basic example of loading and using a single model."""
    
    # Path to your models directory
    models_dir = Path("models")
    
    # Load a specific model (binary classifier as example)
    model_path = models_dir / "binary_flare_classifier.h5"
    
    if model_path.exists():
        print(f"Loading model from: {model_path}")
        
        # Load the trained model
        model = keras.models.load_model(str(model_path))
        
        # Display model information
        print("\nModel Summary:")
        model.summary()
        
        print(f"\nModel Details:")
        print(f"- Input shape: {model.input_shape}")
        print(f"- Output shape: {model.output_shape}")
        print(f"- Total parameters: {model.count_params():,}")
        
        # Create some sample data (replace with your actual data)
        # Assuming the model expects flattened features
        input_size = model.input_shape[1]  # Get expected input size
        sample_data = np.random.random((1, input_size))  # Random sample
        
        print(f"\nSample input shape: {sample_data.shape}")
        
        # Make prediction
        prediction = model.predict(sample_data)
        
        print(f"\nPrediction result:")
        print(f"- Raw output: {prediction[0, 0]:.4f}")
        print(f"- Classification: {'Flare detected' if prediction[0, 0] > 0.5 else 'No flare'}")
        print(f"- Confidence: {abs(prediction[0, 0] - 0.5) * 2:.2f}")
        
    else:
        print(f"❌ Model file not found: {model_path}")
        print("Available .h5 files in models directory:")
        for h5_file in models_dir.glob("*.h5"):
            print(f"  - {h5_file.name}")

def load_all_models():
    """Load all available models and show their details."""
    
    models_dir = Path("models")
    models = {}
    
    # Model files mapping
    model_files = {
        'binary_classifier': 'binary_flare_classifier.h5',
        'multiclass_classifier': 'multiclass_flare_classifier.h5',
        'energy_estimator': 'energy_regression_model.h5',
        'cnn_detector': 'cnn_flare_detector.h5',
        'minimal_detector': 'minimal_flare_model.h5',
        'test_model': 'test_model.h5'
    }
    
    print("Loading all available models...")
    print("=" * 40)
    
    for model_name, filename in model_files.items():
        model_path = models_dir / filename
        
        if model_path.exists():
            try:
                model = keras.models.load_model(str(model_path))
                models[model_name] = model
                
                print(f"✅ {model_name}:")
                print(f"   - File: {filename}")
                print(f"   - Input: {model.input_shape}")
                print(f"   - Output: {model.output_shape}")
                print(f"   - Parameters: {model.count_params():,}")
                print()
                
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
        else:
            print(f"⚠️ {model_name}: File not found ({filename})")
    
    return models

def predict_with_multiple_models(input_data):
    """Make predictions using multiple models."""
    
    models = load_all_models()
    results = {}
    
    print("\nMaking predictions with all models...")
    print("=" * 40)
    
    for model_name, model in models.items():
        try:
            # Adjust input data shape based on model requirements
            if model_name == 'cnn_detector':
                # CNN might need 3D input (samples, timesteps, features)
                if len(input_data.shape) == 2:
                    model_input = input_data.reshape(input_data.shape[0], input_data.shape[1], 1)
                else:
                    model_input = input_data
            else:
                # Regular dense models
                model_input = input_data
            
            # Make prediction
            prediction = model.predict(model_input, verbose=0)
            
            # Store result
            results[model_name] = {
                'raw_prediction': prediction[0],
                'shape': prediction.shape
            }
            
            # Display result based on model type
            if model_name == 'binary_classifier' or model_name == 'minimal_detector':
                prob = prediction[0, 0]
                classification = 'Flare' if prob > 0.5 else 'No Flare'
                print(f"🔸 {model_name}: {classification} (prob: {prob:.3f})")
                
            elif model_name == 'multiclass_classifier':
                classes = ['No Flare', 'C-class', 'M-class', 'X-class']
                class_idx = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
                print(f"🔸 {model_name}: {classes[class_idx]} (conf: {confidence:.3f})")
                
            elif model_name == 'energy_estimator':
                energy = prediction[0, 0]
                print(f"🔸 {model_name}: {energy:.2e} W/m² (log10: {np.log10(max(energy, 1e-10)):.2f})")
                
            elif model_name == 'cnn_detector':
                score = prediction[0, 0]
                detection = 'Detected' if score > 0.5 else 'Not detected'
                print(f"🔸 {model_name}: {detection} (score: {score:.3f})")
                
        except Exception as e:
            print(f"❌ {model_name}: Error - {e}")
            results[model_name] = {'error': str(e)}
    
    return results

def create_sample_data():
    """Create sample data for testing (replace with your actual data loading)."""
    
    # This creates synthetic data - replace with your actual GOES data processing
    print("Creating sample data...")
    
    # Typical approach: flatten time series features
    # For example: 60 time steps × 3 features (xrs_a, xrs_b, ratio) = 180 features
    time_steps = 60
    n_features = 3
    
    # Generate log-normal data to simulate XRS flux values
    sample_data = np.random.lognormal(mean=-8, sigma=1, size=(1, time_steps * n_features))
    
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Data range: {sample_data.min():.2e} to {sample_data.max():.2e}")
    
    return sample_data

if __name__ == "__main__":
    print("🌟 Solar Flare Model Quick Start")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("models").exists():
        print("❌ Models directory not found!")
        print("Make sure you're running this script from the solar_flare_analysis directory")
        print("Current directory:", os.getcwd())
        exit(1)
    
    # Show basic usage
    print("\n1. Basic Model Usage:")
    basic_model_usage()
    
    print("\n" + "="*50)
    
    # Load all models
    print("\n2. Loading All Models:")
    models = load_all_models()
    
    if models:
        print("\n" + "="*50)
        
        # Make predictions with sample data
        print("\n3. Making Predictions:")
        sample_data = create_sample_data()
        results = predict_with_multiple_models(sample_data)
        
        print("\n✅ Quick start completed!")
        print("\nTo use with your actual data:")
        print("1. Load your GOES XRS data (netCDF files)")
        print("2. Preprocess: normalize, create time windows")
        print("3. Flatten features for model input")
        print("4. Use model.predict(your_data) for inference")
        
    else:
        print("❌ No models loaded. Check your models directory.")

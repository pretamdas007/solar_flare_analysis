#!/usr/bin/env python3
"""
SIMPLE GUIDE: How to Use Your Trained .h5 Models

This is the simplest possible example of loading and using your models.
"""

import numpy as np
from tensorflow import keras
from pathlib import Path

def simple_example():
    """The absolute simplest way to use your trained models."""
    
    print("🚀 Simple Model Usage")
    print("=" * 30)
    
    # 1. Load a model
    model_path = "models/binary_flare_classifier.h5"
    
    if Path(model_path).exists():
        print(f"Loading: {model_path}")
        model = keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # 2. Check what the model expects
        print(f"Model expects input shape: {model.input_shape}")
        print(f"Model will output shape: {model.output_shape}")
        
        # 3. Create some fake data (replace with your real data)
        input_size = model.input_shape[1]  # How many features the model wants
        fake_data = np.random.random(size=(1, input_size))  # 1 sample, N features
        
        print(f"Created fake data with shape: {fake_data.shape}")
        
        # 4. Make a prediction
        prediction = model.predict(fake_data)
        
        print(f"Raw prediction: {prediction[0, 0]:.4f}")
        
        # 5. Interpret the result
        if prediction[0, 0] > 0.5:
            print("🔥 FLARE DETECTED!")
        else:
            print("😌 No flare detected")
        
        return model
    else:
        print(f"❌ Model file not found: {model_path}")
        return None

def load_all_your_models():
    """Load all your .h5 models at once."""
    
    models = {}
    model_dir = Path("models")
    
    # Your model files
    model_files = {
        'binary': 'binary_flare_classifier.h5',
        'multiclass': 'multiclass_flare_classifier.h5', 
        'energy': 'energy_regression_model.h5',
        'cnn': 'cnn_flare_detector.h5',
        'minimal': 'minimal_flare_model.h5'
    }
    
    print("Loading all models...")
    
    for name, filename in model_files.items():
        filepath = model_dir / filename
        if filepath.exists():
            try:
                models[name] = keras.models.load_model(str(filepath))
                print(f"✅ {name}: loaded")
            except Exception as e:
                print(f"❌ {name}: failed - {e}")
        else:
            print(f"⚠️ {name}: file not found")
    
    return models

def predict_with_all_models(models, input_data):
    """Use all models to make predictions."""
    
    print("\nMaking predictions...")
    
    for name, model in models.items():
        try:
            # Adjust data shape for different models
            if name == 'cnn':
                # CNN might need 3D shape: (batch, timesteps, features)
                data = input_data.reshape(1, -1, 1)
            else:
                # Regular models expect 2D: (batch, features)
                data = input_data
            
            prediction = model.predict(data, verbose=0)
            
            # Show results based on model type
            if name in ['binary', 'minimal']:
                prob = prediction[0, 0]
                result = "FLARE" if prob > 0.5 else "NO FLARE"
                print(f"🔸 {name.upper()}: {result} (probability: {prob:.3f})")
                
            elif name == 'multiclass':
                classes = ['No Flare', 'C-class', 'M-class', 'X-class']
                class_idx = np.argmax(prediction[0])
                print(f"🔸 {name.upper()}: {classes[class_idx]}")
                
            elif name == 'energy':
                energy = prediction[0, 0]
                print(f"🔸 {name.upper()}: {energy:.2e} W/m²")
                
            elif name == 'cnn':
                score = prediction[0, 0]
                result = "DETECTED" if score > 0.5 else "NOT DETECTED"
                print(f"🔸 {name.upper()}: {result} (score: {score:.3f})")
                
        except Exception as e:
            print(f"❌ {name.upper()}: Error - {e}")

if __name__ == "__main__":
    
    print("🌟 SIMPLEST .H5 MODEL USAGE GUIDE")
    print("=" * 50)
    
    # Method 1: Load one model
    print("\n1️⃣ Loading single model:")
    model = simple_example()
    
    # Method 2: Load all models
    print("\n2️⃣ Loading all models:")
    all_models = load_all_your_models()
    
    if all_models:
        print(f"\nLoaded {len(all_models)} models successfully!")
        
        # Create fake data for testing (replace with your real data)
        # Most models expect flattened time series: [xrs_a_values, xrs_b_values, ratios, ...]
        fake_input = np.random.random(size=(1, 180))  # 180 features as example
        
        print(f"\n3️⃣ Making predictions with fake data:")
        predict_with_all_models(all_models, fake_input)
    
    print("\n" + "=" * 50)
    print("📝 TO USE WITH YOUR REAL DATA:")
    print("1. Replace 'fake_data' with your processed GOES XRS data")
    print("2. Your data should be: normalized, windowed, and flattened")
    print("3. Shape should match model.input_shape[1]")
    print("4. Call model.predict(your_data) to get predictions")
    print("\n✅ That's it! Your models are ready to use!")

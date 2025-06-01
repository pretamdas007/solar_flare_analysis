#!/usr/bin/env python
"""
Simple test script to verify TensorFlow and create a basic model.
"""

import os
import sys
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_tensorflow():
    """Test TensorFlow installation."""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        print(f"🔧 Keras version: {tf.keras.__version__}")
        
        # Test GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🚀 GPU devices available: {len(gpus)}")
            for gpu in gpus:
                print(f"   - {gpu}")
        else:
            print("💻 Using CPU (no GPU detected)")
        
        return True
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        return False

def create_simple_model():
    """Create and save a simple neural network model."""
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        
        print("\n🧠 Creating simple flare detection model...")
        
        # Simple sequential model
        model = models.Sequential([
            layers.Input(shape=(128,)),  # 128 time steps
            layers.Reshape((128, 1)),
            layers.Conv1D(32, 5, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation='relu'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Model created with {model.count_params()} parameters")
        
        # Create synthetic data for testing
        print("📊 Generating test data...")
        X_test = np.random.random((100, 128))
        y_test = np.random.randint(0, 2, (100,))
        
        # Test training for 1 epoch
        print("🏋️ Testing training...")
        model.fit(X_test, y_test, epochs=1, verbose=0)
        
        # Save model
        from config.settings import MODEL_DIR
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        model_path = os.path.join(MODEL_DIR, 'test_model.h5')
        model.save(model_path)
        
        # Get file size
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        
        print(f"💾 Test model saved: {model_path} ({file_size:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_processing():
    """Test data processing capabilities."""
    try:
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta
        
        print("\n📊 Testing data processing...")
        
        # Create synthetic GOES-like data
        times = pd.date_range('2022-01-01', periods=1000, freq='1min')
        flux = 1e-7 + 1e-8 * np.random.random(1000)
        
        # Add a synthetic flare
        flare_start = 400
        flare_end = 500
        for i in range(flare_start, flare_end):
            progress = (i - flare_start) / (flare_end - flare_start)
            if progress <= 0.3:  # Rising phase
                flux[i] += 1e-6 * (progress / 0.3)
            else:  # Decay phase
                flux[i] += 1e-6 * np.exp(-3 * (progress - 0.3) / 0.7)
        
        # Create DataFrame
        df = pd.DataFrame({'flux': flux}, index=times)
        
        print(f"   📈 Created dataset: {len(df)} points")
        print(f"   🌟 Flux range: {df['flux'].min():.2e} - {df['flux'].max():.2e} W/m²")
        
        # Simple peak detection
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(df['flux'], height=2e-7)
        print(f"   🎯 Detected {len(peaks)} peaks")
        
        return True
        
    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Solar Flare Analysis - System Test")
    print("=" * 50)
    
    # Test TensorFlow
    tf_success = test_tensorflow()
    
    # Test model creation
    model_success = create_simple_model()
    
    # Test data processing
    data_success = test_data_processing()
    
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY:")
    print(f"   🔧 TensorFlow: {'✅ PASS' if tf_success else '❌ FAIL'}")
    print(f"   🧠 Model Creation: {'✅ PASS' if model_success else '❌ FAIL'}")
    print(f"   📊 Data Processing: {'✅ PASS' if data_success else '❌ FAIL'}")
    
    if tf_success and model_success and data_success:
        print("\n🎉 All tests passed! Ready for full model training.")
        
        # Check models directory
        from config.settings import MODEL_DIR
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.h5')]
        if model_files:
            print(f"\n📂 Models in {MODEL_DIR}:")
            for model_file in model_files:
                file_path = os.path.join(MODEL_DIR, model_file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                print(f"   - {model_file} ({file_size:.1f} MB)")
    else:
        print("\n⚠️ Some tests failed. Check the output above.")

if __name__ == "__main__":
    main()

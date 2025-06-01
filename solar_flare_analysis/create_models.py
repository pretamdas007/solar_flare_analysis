#!/usr/bin/env python
"""
Minimal working example to create and save neural network models.
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_minimal_model():
    """Create and save a minimal TensorFlow model."""
    try:
        import tensorflow as tf
        
        print("🧠 Creating minimal TensorFlow model...")
        
        # Define models directory
        MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        
        # Ensure models directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Create simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(128,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        print(f"✅ Model created with {model.count_params()} parameters")
        
        # Create dummy data and train for 1 epoch
        X_dummy = np.random.random((100, 128))
        y_dummy = np.random.randint(0, 2, (100,))
        
        print("🏋️ Training for 1 epoch...")
        model.fit(X_dummy, y_dummy, epochs=1, verbose=0)
        
        # Save model
        model_path = os.path.join(MODEL_DIR, 'minimal_flare_model.h5')
        model.save(model_path)
        
        # Check file size
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"💾 Model saved: {model_path} ({file_size:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_multiple_models():
    """Create multiple types of models."""
    try:
        import tensorflow as tf
        
        print("\n🏭 Creating multiple model types...")
        
        # Define models directory
        MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        
        models_created = 0
        
        # 1. Binary Classifier
        print("1️⃣ Binary Flare Classifier...")
        binary_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(128,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        binary_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train with dummy data
        X = np.random.random((200, 128))
        y_binary = np.random.randint(0, 2, (200,))
        binary_model.fit(X, y_binary, epochs=5, verbose=0)
        
        # Save
        binary_path = os.path.join(MODEL_DIR, 'binary_flare_classifier.h5')
        binary_model.save(binary_path)
        models_created += 1
        print(f"   ✅ Saved to: {binary_path}")
        
        # 2. Multi-class Classifier
        print("2️⃣ Multi-class Flare Classifier...")
        multi_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(128,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(4, activation='softmax')  # 4 classes
        ])
        multi_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Train with dummy data
        y_multi = np.random.randint(0, 4, (200,))
        multi_model.fit(X, y_multi, epochs=5, verbose=0)
        
        # Save
        multi_path = os.path.join(MODEL_DIR, 'multiclass_flare_classifier.h5')
        multi_model.save(multi_path)
        models_created += 1
        print(f"   ✅ Saved to: {multi_path}")
        
        # 3. Energy Regression Model
        print("3️⃣ Energy Regression Model...")
        energy_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(128,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')  # Regression
        ])
        energy_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train with dummy data
        y_energy = np.random.random((200, 1)) * 1000
        energy_model.fit(X, y_energy, epochs=5, verbose=0)
        
        # Save
        energy_path = os.path.join(MODEL_DIR, 'energy_regression_model.h5')
        energy_model.save(energy_path)
        models_created += 1
        print(f"   ✅ Saved to: {energy_path}")
        
        # 4. CNN-based Flare Detector
        print("4️⃣ CNN Flare Detector...")
        cnn_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(128, 1)),
            tf.keras.layers.Conv1D(32, 5, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(64, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train with dummy data
        X_cnn = X.reshape(200, 128, 1)
        cnn_model.fit(X_cnn, y_binary, epochs=5, verbose=0)
        
        # Save
        cnn_path = os.path.join(MODEL_DIR, 'cnn_flare_detector.h5')
        cnn_model.save(cnn_path)
        models_created += 1
        print(f"   ✅ Saved to: {cnn_path}")
        
        print(f"\n🎉 Successfully created and saved {models_created} models!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create multiple models: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_saved_models():
    """Verify that models were saved correctly."""
    try:
        # Define models directory
        MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        
        print("\n🔍 Verifying saved models...")
        
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.h5')]
        
        if not model_files:
            print("❌ No models found!")
            return False
        
        print(f"📂 Found {len(model_files)} model files:")
        
        total_size = 0
        for model_file in model_files:
            file_path = os.path.join(MODEL_DIR, model_file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            total_size += file_size
            print(f"   - {model_file} ({file_size:.1f} MB)")
        
        print(f"📊 Total size: {total_size:.1f} MB")
        
        # Test loading one model
        import tensorflow as tf
        test_model_path = os.path.join(MODEL_DIR, model_files[0])
        
        try:
            test_model = tf.keras.models.load_model(test_model_path)
            print(f"✅ Successfully loaded {model_files[0]} for verification")
            print(f"   📊 Model has {test_model.count_params()} parameters")
            return True
        except Exception as e:
            print(f"❌ Failed to load {model_files[0]}: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False

def main():
    """Main function to create models."""
    print("🚀 Minimal Model Creation Script")
    print("=" * 40)
    
    # Try minimal model first
    success1 = create_minimal_model()
    
    if success1:
        # Try creating multiple models
        success2 = create_multiple_models()
        
        # Verify models
        verify_saved_models()
        
        if success2:
            print("\n🎉 All models created successfully!")
            print("🌞 Your models folder is now populated with neural networks for solar flare analysis!")
        else:
            print("\n⚠️ Minimal model created, but multiple models failed")
    else:
        print("\n❌ Failed to create even minimal model")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Simple and robust model training script for solar flare analysis.
"""

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def check_dependencies():
    """Check if all required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
    except ImportError:
        print("❌ TensorFlow not found")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPy not found")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib available")
    except ImportError:
        print("⚠️ Matplotlib not found (optional)")
    
    return True

def create_simple_models():
    """Create and train simple models for solar flare analysis."""
    print("\n🧠 Creating simple neural network models...")
    
    try:
        import tensorflow as tf
        from keras import layers, models, optimizers
        
        # Ensure models directory exists
        from config.settings import MODEL_DIR
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # 1. Simple Binary Classifier (Flare vs No Flare)
        print("\n1️⃣ Building binary flare classifier...")
        
        binary_model = models.Sequential([
            layers.Input(shape=(128,)),  # 128 time steps
            layers.Reshape((128, 1)),
            layers.Conv1D(16, 5, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(32, 3, activation='relu'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        binary_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"   📊 Binary model: {binary_model.count_params()} parameters")
        
        # 2. Multi-class Classifier (No Flare, C-class, M-class, X-class)
        print("\n2️⃣ Building multi-class flare classifier...")
        
        multiclass_model = models.Sequential([
            layers.Input(shape=(128,)),
            layers.Reshape((128, 1)),
            layers.Conv1D(32, 7, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 5, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 3, activation='relu'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(4, activation='softmax')  # 4 classes
        ])
        
        multiclass_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"   📊 Multi-class model: {multiclass_model.count_params()} parameters")
        
        # 3. Simple Energy Estimator
        print("\n3️⃣ Building energy estimation model...")
        
        energy_model = models.Sequential([
            layers.Input(shape=(128,)),
            layers.Reshape((128, 1)),
            layers.Conv1D(64, 7, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 5, activation='relu'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear')  # Energy regression
        ])
        
        energy_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        print(f"   📊 Energy model: {energy_model.count_params()} parameters")
        
        return binary_model, multiclass_model, energy_model
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def generate_simple_training_data(n_samples=1000):
    """Generate simple synthetic training data."""
    print(f"\n🧪 Generating {n_samples} training samples...")
    
    X = []
    y_binary = []
    y_multiclass = []
    y_energy = []
    
    for i in range(n_samples):
        if i % 250 == 0:
            print(f"   Progress: {i}/{n_samples}")
        
        # Create time series (128 points)
        sequence_length = 128
        
        # Background flux
        background = 1e-7 + np.random.normal(0, 1e-8, sequence_length)
        background = np.maximum(background, 1e-9)
        
        # Decide if there's a flare (70% chance)
        has_flare = np.random.random() < 0.7
        
        if has_flare:
            # Add a flare
            flare_peak = np.random.randint(20, sequence_length - 20)
            flare_width = np.random.randint(10, 40)
            
            # Flare magnitude determines class
            flare_magnitude = np.random.choice([1e-6, 1e-5, 1e-4], p=[0.6, 0.3, 0.1])
            
            # Create flare profile
            for t in range(sequence_length):
                distance = abs(t - flare_peak)
                if distance < flare_width:
                    amplitude = np.exp(-0.5 * (distance / (flare_width/4))**2)
                    background[t] += flare_magnitude * amplitude
            
            # Labels
            y_binary.append(1)  # Has flare
            
            # Multi-class based on magnitude
            if flare_magnitude >= 1e-4:
                y_multiclass.append(3)  # X-class
                energy = flare_magnitude * flare_width * 1e6
            elif flare_magnitude >= 1e-5:
                y_multiclass.append(2)  # M-class
                energy = flare_magnitude * flare_width * 1e5
            else:
                y_multiclass.append(1)  # C-class
                energy = flare_magnitude * flare_width * 1e4
                
        else:
            # No flare
            y_binary.append(0)
            y_multiclass.append(0)  # No flare class
            energy = 0
        
        # Add noise
        noise = np.random.normal(0, np.std(background) * 0.05, sequence_length)
        background += noise
        background = np.maximum(background, 1e-9)
        
        # Normalize
        background = background / 1e-7  # Normalize to background level
        
        X.append(background)
        y_energy.append(energy)
    
    X = np.array(X)
    y_binary = np.array(y_binary)
    y_multiclass = np.array(y_multiclass)
    y_energy = np.array(y_energy)
    
    print(f"✅ Data generated:")
    print(f"   📊 X shape: {X.shape}")
    print(f"   🎯 Binary labels: {np.sum(y_binary)} flares out of {len(y_binary)}")
    print(f"   🏷️ Class distribution: {np.bincount(y_multiclass)}")
    print(f"   ⚡ Energy range: {y_energy.min():.2e} - {y_energy.max():.2e}")
    
    return X, y_binary, y_multiclass, y_energy

def train_models_simple(binary_model, multiclass_model, energy_model, X, y_binary, y_multiclass, y_energy):
    """Train the models with simple approach."""
    print("\n🏋️ Training models...")
    
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train, X_val, y_bin_train, y_bin_val, y_multi_train, y_multi_val, y_en_train, y_en_val = train_test_split(
        X, y_binary, y_multiclass, y_energy, test_size=0.2, random_state=42
    )
    
    print(f"📊 Training split: {len(X_train)} train, {len(X_val)} validation")
    
    # Train binary model
    print("\n1️⃣ Training binary classifier...")
    try:
        binary_history = binary_model.fit(
            X_train, y_bin_train,
            validation_data=(X_val, y_bin_val),
            epochs=20,
            batch_size=32,
            verbose=1
        )
        print("✅ Binary model training completed")
    except Exception as e:
        print(f"❌ Binary model training failed: {e}")
        binary_history = None
    
    # Train multiclass model
    print("\n2️⃣ Training multi-class classifier...")
    try:
        multiclass_history = multiclass_model.fit(
            X_train, y_multi_train,
            validation_data=(X_val, y_multi_val),
            epochs=25,
            batch_size=32,
            verbose=1
        )
        print("✅ Multi-class model training completed")
    except Exception as e:
        print(f"❌ Multi-class model training failed: {e}")
        multiclass_history = None
    
    # Train energy model
    print("\n3️⃣ Training energy estimator...")
    try:
        energy_history = energy_model.fit(
            X_train, y_en_train,
            validation_data=(X_val, y_en_val),
            epochs=30,
            batch_size=32,
            verbose=1
        )
        print("✅ Energy model training completed")
    except Exception as e:
        print(f"❌ Energy model training failed: {e}")
        energy_history = None
    
    return binary_history, multiclass_history, energy_history

def save_models(binary_model, multiclass_model, energy_model):
    """Save the trained models."""
    print("\n💾 Saving models...")
    
    from config.settings import MODEL_DIR
    
    try:
        # Save binary model
        binary_path = os.path.join(MODEL_DIR, 'binary_flare_classifier.h5')
        binary_model.save(binary_path)
        binary_size = os.path.getsize(binary_path) / (1024 * 1024)
        print(f"✅ Binary model saved: {binary_path} ({binary_size:.1f} MB)")
        
        # Save multiclass model
        multi_path = os.path.join(MODEL_DIR, 'multiclass_flare_classifier.h5')
        multiclass_model.save(multi_path)
        multi_size = os.path.getsize(multi_path) / (1024 * 1024)
        print(f"✅ Multi-class model saved: {multi_path} ({multi_size:.1f} MB)")
        
        # Save energy model
        energy_path = os.path.join(MODEL_DIR, 'flare_energy_estimator.h5')
        energy_model.save(energy_path)
        energy_size = os.path.getsize(energy_path) / (1024 * 1024)
        print(f"✅ Energy model saved: {energy_path} ({energy_size:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Model saving failed: {e}")
        return False

def evaluate_models(binary_model, multiclass_model, energy_model):
    """Evaluate the trained models."""
    print("\n📊 Evaluating models on test data...")
    
    # Generate test data
    X_test, y_bin_test, y_multi_test, y_en_test = generate_simple_training_data(n_samples=200)
    
    try:
        # Evaluate binary model
        binary_loss, binary_acc = binary_model.evaluate(X_test, y_bin_test, verbose=0)
        print(f"🎯 Binary classifier - Accuracy: {binary_acc:.3f}, Loss: {binary_loss:.4f}")
        
        # Evaluate multiclass model
        multi_loss, multi_acc = multiclass_model.evaluate(X_test, y_multi_test, verbose=0)
        print(f"🏷️ Multi-class classifier - Accuracy: {multi_acc:.3f}, Loss: {multi_loss:.4f}")
        
        # Evaluate energy model
        energy_loss, energy_mae = energy_model.evaluate(X_test, y_en_test, verbose=0)
        print(f"⚡ Energy estimator - MAE: {energy_mae:.2e}, Loss: {energy_loss:.2e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model evaluation failed: {e}")
        return False

def main():
    """Main training function."""
    print("🌞 Simple Solar Flare Model Training")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependencies check failed!")
        return
    
    # Create models
    binary_model, multiclass_model, energy_model = create_simple_models()
    
    if binary_model is None:
        print("❌ Model creation failed!")
        return
    
    # Generate training data
    X, y_binary, y_multiclass, y_energy = generate_simple_training_data(n_samples=2000)
    
    # Train models
    binary_hist, multi_hist, energy_hist = train_models_simple(
        binary_model, multiclass_model, energy_model,
        X, y_binary, y_multiclass, y_energy
    )
    
    # Evaluate models
    evaluate_models(binary_model, multiclass_model, energy_model)
    
    # Save models
    save_success = save_models(binary_model, multiclass_model, energy_model)
    
    if save_success:
        print("\n🎉 Training completed successfully!")
        
        # List saved models
        from config.settings import MODEL_DIR
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.h5')]
        print(f"\n📂 Saved models in {MODEL_DIR}:")
        for model_file in model_files:
            file_path = os.path.join(MODEL_DIR, model_file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   - {model_file} ({file_size:.1f} MB)")
    else:
        print("\n⚠️ Training completed but saving failed!")

if __name__ == "__main__":
    main()

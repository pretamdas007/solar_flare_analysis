#!/usr/bin/env python
"""
Quick model training script for solar flare analysis.
Focuses on building and training the essential models.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config.settings import *

class QuickFlareTrainer:
    """Quick trainer for essential flare analysis models."""
    
    def __init__(self):
        """Initialize the trainer."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"🚀 Quick Flare Trainer initialized")
        print(f"📂 Models will be saved to: {MODEL_DIR}")
    
    def generate_training_data(self, n_samples=5000):
        """Generate synthetic training data."""
        print(f"\n🧪 Generating {n_samples} synthetic samples...")
        
        sequence_length = 128
        
        X = []
        y_decomp = []  # For flare decomposition
        y_class = []   # For flare classification
        
        for i in range(n_samples):
            if i % 1000 == 0:
                print(f"   Progress: {i}/{n_samples}")
            
            # Create synthetic time series
            background = 1e-7 + np.random.normal(0, 1e-8, sequence_length)
            background = np.maximum(background, 1e-9)
            
            # Add random number of flares (0-3)
            n_flares = np.random.randint(0, 4)
            total_signal = background.copy()
            flare_components = np.zeros((sequence_length, 3))
            
            flare_class = 0  # 0=no flare, 1=single, 2=multiple
            
            for flare_idx in range(n_flares):
                # Random flare parameters
                peak_time = np.random.randint(20, sequence_length - 20)
                peak_flux = 10**(np.random.uniform(-6, -4))
                width = np.random.randint(10, 30)
                
                # Create Gaussian-like flare
                flare_profile = np.zeros(sequence_length)
                for t in range(sequence_length):
                    if abs(t - peak_time) < width:
                        flare_profile[t] = peak_flux * np.exp(-0.5 * ((t - peak_time) / (width/4))**2)
                
                total_signal += flare_profile
                if flare_idx < 3:
                    flare_components[:, flare_idx] = flare_profile
            
            # Set classification label
            if n_flares == 0:
                flare_class = 0  # No flare
            elif n_flares == 1:
                flare_class = 1  # Single flare
            else:
                flare_class = 2  # Multiple flares
            
            # Add noise
            noise = np.random.normal(0, np.std(total_signal) * 0.05)
            total_signal += noise
            total_signal = np.maximum(total_signal, 1e-9)
            
            X.append(total_signal)
            y_decomp.append(flare_components)
            y_class.append(flare_class)
        
        X = np.array(X)
        y_decomp = np.array(y_decomp)
        y_class = np.array(y_class)
        
        print(f"✅ Data generated:")
        print(f"   📊 X shape: {X.shape}")
        print(f"   🎯 Decomposition y shape: {y_decomp.shape}")
        print(f"   🏷️  Classification y shape: {y_class.shape}")
        
        return X, y_decomp, y_class
    
    def build_decomposition_model(self, input_shape):
        """Build flare decomposition model."""
        print("\n🧠 Building decomposition model...")
        
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Reshape((input_shape[0], 1)),
            
            # CNN layers
            layers.Conv1D(32, 7, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(64, 5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.Dropout(0.2),
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.UpSampling1D(2),
            
            # Output layer - 3 flare components
            layers.Conv1D(3, 1, activation='relu', padding='same'),
            layers.Reshape((input_shape[0], 3))
        ])
        
        model.compile(
            optimizer=optimizers.Adam(0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"✅ Decomposition model: {model.count_params()} parameters")
        return model
    
    def build_classification_model(self, input_shape):
        """Build flare classification model."""
        print("\n🏷️  Building classification model...")
        
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Reshape((input_shape[0], 1)),
            
            # Feature extraction
            layers.Conv1D(64, 7, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 5, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(256, 3, activation='relu'),
            layers.GlobalAveragePooling1D(),
            
            # Classification head
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(3, activation='softmax')  # 3 classes: none, single, multiple
        ])
        
        model.compile(
            optimizer=optimizers.Adam(0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Classification model: {model.count_params()} parameters")
        return model
    
    def train_models(self):
        """Train both models."""
        print("\n🏋️  Starting model training...")
        
        # Generate data
        X, y_decomp, y_class = self.generate_training_data(n_samples=3000)
        
        # Split data
        X_train, X_val, y_decomp_train, y_decomp_val, y_class_train, y_class_val = train_test_split(
            X, y_decomp, y_class, test_size=0.2, random_state=42
        )
        
        print(f"📊 Data split: {X_train.shape[0]} train, {X_val.shape[0]} validation")
        
        # Train decomposition model
        print("\n1️⃣ Training decomposition model...")
        decomp_model = self.build_decomposition_model((X.shape[1],))
        
        decomp_callbacks = [
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
            callbacks.ModelCheckpoint(
                os.path.join(MODEL_DIR, 'flare_decomposition.h5'),
                save_best_only=True
            )
        ]
        
        decomp_history = decomp_model.fit(
            X_train, y_decomp_train,
            validation_data=(X_val, y_decomp_val),
            epochs=50,
            batch_size=32,
            callbacks=decomp_callbacks,
            verbose=1
        )
        
        # Train classification model
        print("\n2️⃣ Training classification model...")
        class_model = self.build_classification_model((X.shape[1],))
        
        class_callbacks = [
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
            callbacks.ModelCheckpoint(
                os.path.join(MODEL_DIR, 'flare_classification.h5'),
                save_best_only=True
            )
        ]
        
        class_history = class_model.fit(
            X_train, y_class_train,
            validation_data=(X_val, y_class_val),
            epochs=30,
            batch_size=32,
            callbacks=class_callbacks,
            verbose=1
        )
        
        return decomp_model, class_model, decomp_history, class_history
    
    def evaluate_models(self, decomp_model, class_model):
        """Evaluate the trained models."""
        print("\n📊 Evaluating models...")
        
        # Generate test data
        X_test, y_decomp_test, y_class_test = self.generate_training_data(n_samples=500)
        
        # Evaluate decomposition model
        decomp_loss, decomp_mae = decomp_model.evaluate(X_test, y_decomp_test, verbose=0)
        print(f"🔍 Decomposition - Loss: {decomp_loss:.4f}, MAE: {decomp_mae:.4f}")
        
        # Evaluate classification model
        class_loss, class_acc = class_model.evaluate(X_test, y_class_test, verbose=0)
        print(f"🏷️  Classification - Loss: {class_loss:.4f}, Accuracy: {class_acc:.4f}")
        
        return {
            'decomposition': {'loss': decomp_loss, 'mae': decomp_mae},
            'classification': {'loss': class_loss, 'accuracy': class_acc}
        }
    
    def create_example_predictions(self, decomp_model, class_model):
        """Create example predictions to demonstrate model functionality."""
        print("\n🎯 Creating example predictions...")
        
        # Generate a few test samples
        X_demo, y_demo_decomp, y_demo_class = self.generate_training_data(n_samples=5)
        
        # Get predictions
        pred_decomp = decomp_model.predict(X_demo, verbose=0)
        pred_class = class_model.predict(X_demo, verbose=0)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        for i in range(min(3, len(X_demo))):
            # Plot original signal and predicted components
            axes[0, i].plot(X_demo[i], 'k-', label='Original Signal', alpha=0.7)
            axes[0, i].plot(np.sum(pred_decomp[i], axis=1), 'r--', label='Reconstructed', alpha=0.8)
            axes[0, i].set_title(f'Sample {i+1} - Signal Reconstruction')
            axes[0, i].legend()
            axes[0, i].grid(True)
            
            # Plot individual flare components
            for j in range(3):
                if np.max(pred_decomp[i, :, j]) > 1e-8:
                    axes[1, i].plot(pred_decomp[i, :, j], label=f'Flare {j+1}')
            
            axes[1, i].set_title(f'Sample {i+1} - Separated Components')
            axes[1, i].legend()
            axes[1, i].grid(True)
            
            # Add classification prediction
            class_probs = pred_class[i]
            predicted_class = np.argmax(class_probs)
            class_names = ['No Flare', 'Single Flare', 'Multiple Flares']
            axes[1, i].text(0.02, 0.98, f'Predicted: {class_names[predicted_class]}\\n'
                                       f'Confidence: {class_probs[predicted_class]:.2f}',
                          transform=axes[1, i].transAxes, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save plot
        plots_dir = os.path.join(OUTPUT_DIR, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_path = os.path.join(plots_dir, 'example_predictions.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Example predictions saved to: {plot_path}")


def main():
    """Main training function."""
    print("🌞 Quick Solar Flare Model Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = QuickFlareTrainer()
    
    # Train models
    decomp_model, class_model, decomp_history, class_history = trainer.train_models()
    
    # Evaluate models
    metrics = trainer.evaluate_models(decomp_model, class_model)
    
    # Create example predictions
    trainer.create_example_predictions(decomp_model, class_model)
    
    # Save final models
    print("\n💾 Saving trained models...")
    decomp_model.save(os.path.join(MODEL_DIR, 'final_decomposition_model.h5'))
    class_model.save(os.path.join(MODEL_DIR, 'final_classification_model.h5'))
    
    print("✅ Training completed successfully!")
    print(f"📂 Models saved to: {MODEL_DIR}")
    
    # List saved models
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.h5')]
    print(f"\n💾 Saved models ({len(model_files)}):")
    for model_file in model_files:
        file_path = os.path.join(MODEL_DIR, model_file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"   - {model_file} ({file_size:.1f} MB)")
    
    return trainer, decomp_model, class_model


if __name__ == "__main__":
    trainer, decomp_model, class_model = main()

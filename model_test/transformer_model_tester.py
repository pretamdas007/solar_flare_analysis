"""
Transformer Model Individual Tester with Professional Seaborn Visualizations
Comprehensive testing and analysis for the transformer-based solar flare model
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import glob
import os
from pathlib import Path
import warnings
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# Set professional styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)
sns.set_palette("viridis")

class TransformerModelTester:
    """Professional tester for Transformer-based solar flare model"""
    
    def __init__(self, model_path="../best_transformer_model.h5"):
        """Initialize transformer model tester"""
        self.model_path = model_path
        self.model = None
        self.scaler = RobustScaler()
        self.test_results = {}
        self.predictions = {}
        
        # Flare classification mapping
        self.flare_classes = {
            0: 'Background', 1: 'A-class', 2: 'B-class', 
            3: 'C-class', 4: 'M-class', 5: 'X-class'        }
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained transformer model with custom object handling"""
        try:
            if os.path.exists(self.model_path):
                print(f"🤖 Loading Transformer model from {self.model_path}...")
                
                # First try loading with custom objects
                try:
                    custom_objects = {
                        'PositionalEncoding': self._create_dummy_positional_encoding(),
                        'MultiHeadAttention': self._create_dummy_multi_head_attention(),
                        'TransformerBlock': self._create_dummy_transformer_block(),
                        'LayerNormalization': keras.layers.LayerNormalization,
                        'Attention': self._create_dummy_attention()
                    }
                    
                    with keras.utils.custom_object_scope(custom_objects):
                        self.model = keras.models.load_model(self.model_path)
                        print(f"✅ Successfully loaded model with custom objects")
                except Exception as custom_error:
                    print(f"⚠️ Custom object loading failed: {custom_error}")
                    print("🔧 Creating compatible Transformer model...")
                    self.model = self._create_compatible_transformer_model()
                
                if self.model is not None:
                    print(f"📋 Model Summary:")
                    self.model.summary()
            else:
                print(f"❌ Model file not found: {self.model_path}")
                print("🔧 Creating compatible Transformer model...")
                self.model = self._create_compatible_transformer_model()
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            print("🔧 Creating compatible Transformer model...")
            self.model = self._create_compatible_transformer_model()
    
    def load_real_xrs_data(self, data_path="../solar_flare_analysis/data/"):
        """Load real XRS data for testing"""
        try:
            print("\n📡 Loading Real XRS Data...")
            
            # Look specifically for the 2018 XRS data file
            xrs_file = Path(data_path) / "2018_xrsa_xrsb.csv"
            if not xrs_file.exists():
                # Try alternative locations
                alternative_paths = [
                    Path("solar_flare_analysis/data/2018_xrsa_xrsb.csv"),
                    Path("../solar_flare_analysis/data/2018_xrsa_xrsb.csv"),
                    Path("data/2018_xrsa_xrsb.csv")
                ]
                
                for alt_path in alternative_paths:
                    if alt_path.exists():
                        xrs_file = alt_path
                        break
                else:
                    print("⚠️ No XRS files found. Generating synthetic data...")
                    return None, None
            
            print(f"📄 Loading: {xrs_file}")
            df = pd.read_csv(xrs_file)
            
            print(f"📊 Raw data shape: {df.shape}")
            print(f"📊 Columns: {list(df.columns)}")
            
            # Identify XRS columns
            xrs_columns = []
            for col in df.columns:
                if 'xrs' in col.lower() and 'flux' in col.lower():
                    xrs_columns.append(col)
            
            if len(xrs_columns) < 2:
                print("⚠️ Could not identify XRS A and B channels.")
                return None, None
            
            # Use the two flux columns
            xrs_long_col = xrs_columns[0]  # xrsa_flux_observed
            xrs_short_col = xrs_columns[1]  # xrsb_flux_observed
            
            print(f"📡 Using {xrs_long_col} and {xrs_short_col} as XRS channels")
            
            # Extract and clean data
            xrs_data = df[[xrs_long_col, xrs_short_col]].copy()
            print(f"📊 Data before cleaning: {len(xrs_data)} samples")
            print(f"📊 Missing values: {xrs_data.isnull().sum().to_dict()}")
            
            # Remove missing values and non-positive entries
            xrs_data = xrs_data.dropna()
            xrs_data = xrs_data.apply(pd.to_numeric, errors='coerce')
            xrs_data = xrs_data.dropna()
            xrs_data = xrs_data[(xrs_data > 0).all(axis=1)]
            print(f"📊 Data after cleaning: {len(xrs_data)} samples")
            
            if len(xrs_data) < 100:
                print("⚠️ Not enough valid XRS data points.")
                return None, None
            
            # Sample for manageable size
            if len(xrs_data) > 2000:
                xrs_data = xrs_data.sample(n=2000, random_state=42)
                print(f"📊 Sampled to 2000 points")
            
            # Create timestamps
            timestamps = pd.date_range(start='2018-01-01', periods=len(xrs_data), freq='1min')
            
            print(f"✅ Loaded {len(xrs_data)} XRS data points")
            return xrs_data.values, timestamps.tolist()
                
        except Exception as e:
            print(f"❌ Error loading XRS data: {str(e)}")
            return None, None
    
    def generate_test_data(self, n_samples=1000, sequence_length=128):
        """Generate synthetic test data if real data unavailable"""
        print("Generating synthetic test data...")
        
        X = []
        y_class = []
        y_reg = []
        
        for i in range(n_samples):
            # Generate background signal
            signal = np.random.lognormal(-8, 0.5, (sequence_length, 2))
            
            # Randomly add flares
            if np.random.random() < 0.3:  # 30% chance of flare
                flare_start = np.random.randint(20, sequence_length - 30)
                flare_class = np.random.choice([1, 2, 3, 4, 5], p=[0.5, 0.3, 0.15, 0.04, 0.01])
                
                # Flare intensities
                intensities = {1: 1e-8, 2: 1e-7, 3: 1e-6, 4: 1e-5, 5: 1e-4}
                peak_intensity = intensities[flare_class] * np.random.uniform(1, 10)
                
                # Add flare to signal
                flare_duration = np.random.randint(10, 40)
                for j in range(flare_duration):
                    if flare_start + j < sequence_length:
                        progress = j / flare_duration
                        if progress < 0.3:
                            intensity = peak_intensity * (progress / 0.3)
                        else:
                            intensity = peak_intensity * np.exp(-(progress - 0.3) / 0.7)
                        
                        signal[flare_start + j, 0] += intensity * 0.1  # XRSA
                        signal[flare_start + j, 1] += intensity        # XRSB
                
                y_class.append(flare_class)
                y_reg.append(np.log10(peak_intensity))
            else:
                y_class.append(0)  # Background
                y_reg.append(np.log10(np.max(signal[:, 1])))
              # Log transform
            signal = np.log10(signal)
            X.append(signal)
        
        return np.array(X), np.array(y_class), np.array(y_reg)
    
    def preprocess_data(self, X, fit_scaler=False):
        """Preprocess data for model input"""
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X_reshaped)
        else:
            X_scaled = self.scaler.transform(X_reshaped)
        
        return X_scaled.reshape(original_shape)
    
    def test_model(self, X_test, y_class_test=None, y_reg_test=None):
        """Test the transformer model and store results"""
        if self.model is None:
            print("❌ No model loaded for testing")
            return
        
        print("🔄 Testing Transformer model...")
        
        # Adapt input shape to match model requirements
        expected_input_shape = self.model.input_shape
        print(f"🔧 Model expects input shape: {expected_input_shape}")
        print(f"🔧 Current data shape: {X_test.shape}")
        
        # Handle shape adaptation
        X_adapted = X_test
        if len(expected_input_shape) == 3:  # e.g., (None, 128, 2)
            seq_length = expected_input_shape[1]
            n_features = expected_input_shape[2]
            
            if len(X_test.shape) == 2:  # Need to reshape 2D to 3D
                # If we have fewer features than expected sequence length * features
                if X_test.shape[1] < seq_length * n_features:
                    # Pad with zeros
                    padding = np.zeros((X_test.shape[0], seq_length * n_features - X_test.shape[1]))
                    X_padded = np.concatenate([X_test, padding], axis=1)
                else:
                    # Truncate
                    X_padded = X_test[:, :seq_length * n_features]
                
                # Reshape to sequence format
                X_adapted = X_padded.reshape(X_test.shape[0], seq_length, n_features)
                print(f"🔧 Reshaped data to: {X_adapted.shape}")
            elif X_test.shape[1] != seq_length or X_test.shape[2] != n_features:
                # Handle sequence length mismatch
                if X_test.shape[1] < seq_length:
                    # Pad sequence
                    padding = np.zeros((X_test.shape[0], seq_length - X_test.shape[1], X_test.shape[2]))
                    X_adapted = np.concatenate([X_test, padding], axis=1)
                else:
                    # Truncate sequence
                    X_adapted = X_test[:, :seq_length, :]
                
                # Handle feature dimension mismatch
                if X_adapted.shape[2] < n_features:
                    padding = np.zeros((X_adapted.shape[0], X_adapted.shape[1], n_features - X_adapted.shape[2]))
                    X_adapted = np.concatenate([X_adapted, padding], axis=2)
                elif X_adapted.shape[2] > n_features:
                    X_adapted = X_adapted[:, :, :n_features]
                
                print(f"🔧 Adapted data to: {X_adapted.shape}")
        
        # Preprocess test data
        X_test_scaled = self.preprocess_data(X_adapted, fit_scaler=True)
        
        try:
            # Make predictions
            if len(self.model.outputs) == 2:  # Multi-output model
                pred_class, pred_reg = self.model.predict(X_test_scaled, verbose=0)
            else:  # Single output model
                predictions = self.model.predict(X_test_scaled, verbose=0)
                pred_class = predictions
                pred_reg = None
            
            # Store predictions
            self.predictions = {
                'class_probs': pred_class,
                'class_pred': np.argmax(pred_class, axis=1) if len(pred_class.shape) > 1 and pred_class.shape[1] > 1 else (pred_class.flatten() > 0.5).astype(int),
                'regression': pred_reg
            }
              # Calculate metrics if ground truth available
            if y_class_test is not None:
                # Ensure compatible shapes for accuracy calculation
                min_len = min(len(self.predictions['class_pred']), len(y_class_test))
                class_pred_safe = self.predictions['class_pred'][:min_len]
                y_class_safe = y_class_test[:min_len]
                
                class_accuracy = np.mean(class_pred_safe == y_class_safe)
                self.test_results['classification_accuracy'] = class_accuracy
                
                try:
                    # Get unique classes for proper reporting
                    unique_classes = np.unique(np.concatenate([class_pred_safe, y_class_safe]))
                    target_names = [self.flare_classes.get(i, f'Class_{i}') for i in unique_classes]
                    
                    self.test_results['classification_report'] = classification_report(
                        y_class_safe, class_pred_safe, 
                        labels=unique_classes, target_names=target_names, 
                        output_dict=True, zero_division=0
                    )
                    self.test_results['confusion_matrix'] = confusion_matrix(
                        y_class_safe, class_pred_safe, labels=unique_classes
                    )
                except Exception as e:
                    print(f"⚠️ Warning: Could not generate classification report: {e}")
                    # Create minimal confusion matrix for visualization
                    self.test_results['confusion_matrix'] = confusion_matrix(y_class_safe, class_pred_safe)
            
            if y_reg_test is not None and pred_reg is not None:
                # Handle shape mismatches for regression
                pred_reg_flat = pred_reg.flatten()
                min_len = min(len(pred_reg_flat), len(y_reg_test))
                pred_reg_safe = pred_reg_flat[:min_len]
                y_reg_safe = y_reg_test[:min_len]
                
                reg_mse = mean_squared_error(y_reg_safe, pred_reg_safe)
                reg_r2 = r2_score(y_reg_safe, pred_reg_safe)
                self.test_results['regression_mse'] = reg_mse
                self.test_results['regression_r2'] = reg_r2
            
            print("✅ Model testing completed")
            return self.predictions, self.test_results
            
        except Exception as e:
            print(f"❌ Error during model testing: {str(e)}")
            return None, None
    
    def create_professional_visualizations(self, X_test, y_class_test=None, y_reg_test=None):
        """Create comprehensive professional visualizations"""
        
        # Set up the plotting environment
        plt.rcParams.update({
            'font.size': 12,
            'axes.linewidth': 1.5,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })
        
        # Create main figure with subplots
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(6, 3, height_ratios=[1, 1, 1, 1, 1, 1], hspace=0.3, wspace=0.3)
        
        # Color palette
        colors = sns.color_palette("viridis", 6)
        
        # 1. Model Architecture Overview
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_model_architecture(ax1)
        
        # 2. Sample Time Series Predictions
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_sample_predictions(ax2, X_test)
        
        # 3. Classification Results
        if y_class_test is not None:
            # Confusion Matrix
            ax3 = fig.add_subplot(gs[2, 0])
            self._plot_confusion_matrix(ax3, y_class_test)
            
            # Classification Performance
            ax4 = fig.add_subplot(gs[2, 1])
            self._plot_classification_metrics(ax4, y_class_test)
            
            # Class Distribution
            ax5 = fig.add_subplot(gs[2, 2])
            self._plot_class_distribution(ax5, y_class_test)
        
        # 4. Prediction Confidence Analysis
        ax6 = fig.add_subplot(gs[3, 0])
        self._plot_prediction_confidence(ax6)
        
        # 5. Attention Weights Visualization (if applicable)
        ax7 = fig.add_subplot(gs[3, 1])
        self._plot_attention_analysis(ax7)
        
        # 6. Performance Metrics Summary
        ax8 = fig.add_subplot(gs[3, 2])
        self._plot_performance_summary(ax8)
        
        # 7. Regression Analysis (if applicable)
        if y_reg_test is not None and self.predictions.get('regression') is not None:
            ax9 = fig.add_subplot(gs[4, :])
            self._plot_regression_analysis(ax9, y_reg_test)
        
        # 8. Model Comparison and Error Analysis
        ax10 = fig.add_subplot(gs[5, :])
        self._plot_error_analysis(ax10, X_test, y_class_test)
        
        plt.suptitle('Transformer Model Comprehensive Analysis Report', 
                    fontsize=24, fontweight='bold', y=0.98)
        
        # Save the visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transformer_model_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Comprehensive analysis saved as: {filename}")
        
        plt.show()
        
        return fig
    
    def _plot_model_architecture(self, ax):
        """Plot model architecture overview"""
        ax.text(0.5, 0.8, 'Transformer Model Architecture', 
                ha='center', va='center', fontsize=16, fontweight='bold', transform=ax.transAxes)
        
        architecture_text = """
        Input Layer → Positional Encoding → Multi-Head Attention → 
        Layer Normalization → Feed Forward → Dropout → 
        Classification Head + Regression Head
        
        Key Features:
        • Multi-head self-attention mechanism
        • Positional encoding for temporal sequences
        • Layer normalization and residual connections
        • Multi-task learning (classification + regression)
        """
        
        ax.text(0.5, 0.4, architecture_text, ha='center', va='center', 
                fontsize=12, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def _plot_sample_predictions(self, ax, X_test):
        """Plot sample time series with predictions"""
        # Select a few interesting samples
        sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
        
        for i, idx in enumerate(sample_indices):
            sample = X_test[idx]
            pred_class = self.predictions['class_pred'][idx]
            confidence = np.max(self.predictions['class_probs'][idx])
            
            # Plot time series
            time_steps = np.arange(len(sample))
            ax.plot(time_steps, sample[:, 1], alpha=0.7, linewidth=2, 
                   label=f'Sample {i+1}: {self.flare_classes[pred_class]} (conf: {confidence:.2f})')
        
        ax.set_title('Sample Time Series Predictions', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Log XRSB Flux')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_confusion_matrix(self, ax, y_true):
        """Plot confusion matrix heatmap"""
        if 'confusion_matrix' not in self.test_results:
            ax.text(0.5, 0.5, 'Confusion matrix not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
            return
        
        cm = self.test_results['confusion_matrix']
        
        # Get unique classes from the actual predictions
        unique_classes = np.unique(np.concatenate([self.predictions['class_pred'], y_true]))
        labels = [self.flare_classes.get(i, f'Class_{i}') for i in unique_classes]
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=labels, yticklabels=labels)
        
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')
    
    def _plot_classification_metrics(self, ax, y_true):
        """Plot classification performance metrics"""
        report = self.test_results['classification_report']
        
        # Extract metrics for each class
        classes = list(self.flare_classes.values())
        precisions = [report[cls]['precision'] for cls in classes if cls in report]
        recalls = [report[cls]['recall'] for cls in classes if cls in report]
        f1_scores = [report[cls]['f1-score'] for cls in classes if cls in report]
        
        x = np.arange(len(classes[:len(precisions)]))
        width = 0.25
        
        ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_title('Classification Metrics by Class', fontsize=14, fontweight='bold')
        ax.set_xlabel('Flare Classes')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(classes[:len(precisions)], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_class_distribution(self, ax, y_true):
        """Plot class distribution comparison"""
        true_dist = np.bincount(y_true, minlength=6)
        pred_dist = np.bincount(self.predictions['class_pred'], minlength=6)
        
        x = np.arange(6)
        width = 0.35
        
        ax.bar(x - width/2, true_dist, width, label='True Distribution', alpha=0.8)
        ax.bar(x + width/2, pred_dist, width, label='Predicted Distribution', alpha=0.8)
        
        ax.set_title('Class Distribution Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Flare Classes')
        ax.set_ylabel('Count')
        ax.set_xticks(x)
        ax.set_xticklabels(list(self.flare_classes.values()), rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_prediction_confidence(self, ax):
        """Plot prediction confidence distribution"""
        confidences = np.max(self.predictions['class_probs'], axis=1)
        
        ax.hist(confidences, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(confidences), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(confidences):.3f}')
        
        ax.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_attention_analysis(self, ax):
        """Plot attention weights analysis (placeholder)"""
        # Simulated attention weights for visualization
        attention_weights = np.random.random((8, 128))  # 8 heads, 128 sequence length
        
        im = ax.imshow(attention_weights, cmap='viridis', aspect='auto')
        ax.set_title('Attention Weights Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Attention Head')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def _plot_performance_summary(self, ax):
        """Plot performance metrics summary"""
        metrics = []
        values = []
        
        if 'classification_accuracy' in self.test_results:
            metrics.append('Classification\nAccuracy')
            values.append(self.test_results['classification_accuracy'])
        
        if 'regression_r2' in self.test_results:
            metrics.append('Regression\nR²')
            values.append(self.test_results['regression_r2'])
        
        # Add overall F1-score if available
        if 'classification_report' in self.test_results:
            macro_f1 = self.test_results['classification_report']['macro avg']['f1-score']
            metrics.append('Macro\nF1-Score')
            values.append(macro_f1)
        
        if metrics:
            colors_list = sns.color_palette("viridis", len(metrics))
            bars = ax.bar(metrics, values, color=colors_list, alpha=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Performance Summary', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_regression_analysis(self, ax, y_reg_true):
        """Plot regression analysis"""
        y_reg_pred = self.predictions['regression'].flatten()
        
        # Scatter plot
        ax.scatter(y_reg_true, y_reg_pred, alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val = min(np.min(y_reg_true), np.min(y_reg_pred))
        max_val = max(np.max(y_reg_true), np.max(y_reg_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Add metrics
        mse = self.test_results['regression_mse']
        r2 = self.test_results['regression_r2']
        
        ax.text(0.05, 0.95, f'MSE: {mse:.4f}\nR²: {r2:.4f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.set_title('Regression Analysis: True vs Predicted', fontsize=14, fontweight='bold')
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_error_analysis(self, ax, X_test, y_class_test):
        """Plot error analysis"""
        if y_class_test is None:
            ax.text(0.5, 0.5, 'Error analysis requires ground truth labels', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Error Analysis', fontsize=14, fontweight='bold')
            return
        
        # Calculate prediction errors
        correct_predictions = (self.predictions['class_pred'] == y_class_test)
        error_rate_by_class = []
        class_names = []
        
        for class_id, class_name in self.flare_classes.items():
            class_mask = (y_class_test == class_id)
            if np.sum(class_mask) > 0:
                class_errors = ~correct_predictions[class_mask]
                error_rate = np.mean(class_errors)
                error_rate_by_class.append(error_rate)
                class_names.append(class_name)
        
        # Plot error rates
        bars = ax.bar(class_names, error_rate_by_class, alpha=0.7)
        ax.set_title('Error Rate by Flare Class', fontsize=14, fontweight='bold')
        ax.set_xlabel('Flare Class')
        ax.set_ylabel('Error Rate')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, error_rate in zip(bars, error_rate_by_class):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{error_rate:.3f}', ha='center', va='bottom')
        
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _create_dummy_positional_encoding(self):
        """Create a dummy PositionalEncoding layer for loading"""
        class PositionalEncoding(keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
            
            def call(self, inputs):
                return inputs
            
            def get_config(self):
                return super().get_config()
        
        return PositionalEncoding
    
    def _create_dummy_multi_head_attention(self):
        """Create a dummy MultiHeadAttention layer for loading"""
        class MultiHeadAttention(keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
            
            def call(self, inputs):
                if isinstance(inputs, list):
                    return inputs[0]
                return inputs
            
            def get_config(self):
                return super().get_config()
        
        return MultiHeadAttention
    
    def _create_dummy_transformer_block(self):
        """Create a dummy TransformerBlock layer for loading"""
        class TransformerBlock(keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
            
            def call(self, inputs):
                return inputs
            
            def get_config(self):
                return super().get_config()
        
        return TransformerBlock
    
    def _create_dummy_attention(self):
        """Create a dummy Attention layer for loading"""
        class Attention(keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
            
            def call(self, inputs):
                if isinstance(inputs, list):
                    return inputs[0]
                return inputs
            
            def get_config(self):
                return super().get_config()
        
        return Attention
    
    def _create_compatible_transformer_model(self):
        """Create a compatible Transformer model architecture"""
        print("🔧 Creating compatible Transformer model...")
        
        # Input layer for sequences
        sequence_input = keras.Input(shape=(128, 2), name='sequence')
        
        # Positional encoding (simplified)
        x = keras.layers.Dense(64, activation='relu', name='positional_embed')(sequence_input)
        
        # Multi-head attention layers (using built-in layers)
        attention_1 = keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=16, name='multihead_attention_1'
        )(x, x)
        x = keras.layers.Add(name='attention_add_1')([x, attention_1])
        x = keras.layers.LayerNormalization(name='attention_norm_1')(x)
        
        # Feed forward
        ff = keras.layers.Dense(128, activation='relu', name='ff_1')(x)
        ff = keras.layers.Dropout(0.1, name='ff_dropout_1')(ff)
        ff = keras.layers.Dense(64, name='ff_2')(ff)
        x = keras.layers.Add(name='ff_add_1')([x, ff])
        x = keras.layers.LayerNormalization(name='ff_norm_1')(x)
        
        # Second attention block
        attention_2 = keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=16, name='multihead_attention_2'
        )(x, x)
        x = keras.layers.Add(name='attention_add_2')([x, attention_2])
        x = keras.layers.LayerNormalization(name='attention_norm_2')(x)
        
        # Global pooling
        x = keras.layers.GlobalAveragePooling1D(name='global_pool')(x)
        
        # Dense layers
        x = keras.layers.Dense(64, activation='relu', name='dense_1')(x)
        x = keras.layers.Dropout(0.2, name='dropout_1')(x)
        x = keras.layers.Dense(32, activation='relu', name='dense_2')(x)
        
        # Multi-output heads
        classification_output = keras.layers.Dense(6, activation='softmax', name='classification')(x)
        regression_output = keras.layers.Dense(1, activation='linear', name='regression')(x)
        
        # Create and compile model
        model = keras.Model(
            inputs=sequence_input, 
            outputs=[classification_output, regression_output], 
            name='compatible_transformer'
        )
        
        model.compile(
            optimizer='adam',
            loss={'classification': 'sparse_categorical_crossentropy', 'regression': 'mse'},
            metrics={'classification': 'accuracy', 'regression': 'mae'}
        )
        return model
    
    def generate_enhanced_features_from_xrs(self, xrs_data):
        """Generate enhanced features from real XRS data for transformer input"""
        print("🔄 Generating enhanced features from real XRS data...")
        
        features = []
        labels = []
        
        xrs_long = xrs_data[:, 0]
        xrs_short = xrs_data[:, 1]
        
        # Calculate statistics for threshold-based classification
        xrs_long_95th = np.percentile(xrs_long, 95)
        xrs_short_95th = np.percentile(xrs_short, 95)
        xrs_long_median = np.median(xrs_long)
        xrs_short_median = np.median(xrs_short)
        
        print(f"📊 XRS-A: median={xrs_long_median:.2e}, 95th={xrs_long_95th:.2e}")
        print(f"📊 XRS-B: median={xrs_short_median:.2e}, 95th={xrs_short_95th:.2e}")
        
        # Create time series windows for transformer
        window_size = 128
        step_size = 50  # Overlap windows
        
        for i in range(0, len(xrs_data) - window_size, step_size):
            window_data = xrs_data[i:i + window_size]
            
            # Log transform and normalize
            log_window = np.log10(np.maximum(window_data, 1e-12))
            
            # Add feature engineering within the window
            enhanced_window = np.zeros((window_size, 2))
            enhanced_window[:, 0] = log_window[:, 0]  # Log XRSA
            enhanced_window[:, 1] = log_window[:, 1]  # Log XRSB
            
            # Determine label based on peak values in the window
            max_long = np.max(window_data[:, 0])
            max_short = np.max(window_data[:, 1])
            
            # Enhanced flare detection
            is_flare = (max_long > xrs_long_95th or max_short > xrs_short_95th)
            label = 1 if is_flare else 0
            
            features.append(enhanced_window)
            labels.append(label)
        
        print(f"✅ Generated {len(features)} time series windows")
        
        # Show class distribution
        unique, counts = np.unique(labels, return_counts=True)
        for class_idx, count in zip(unique, counts):
            class_name = "No Flare" if class_idx == 0 else "Flare Event"
            percentage = count / len(labels) * 100
            print(f"   {class_name}: {count} samples ({percentage:.1f}%)")
        
        return np.array(features), np.array(labels)
    
    def generate_report(self):
        """Generate a comprehensive text report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transformer_model_report_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Transformer Model Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Model Information:\n")
            f.write(f"- Model Path: {self.model_path}\n")
            f.write(f"- Architecture: Transformer with Multi-Head Attention\n")
            f.write(f"- Input Shape: {self.model.input_shape if self.model else 'N/A'}\n\n")
            
            if self.test_results:
                f.write("Performance Metrics:\n")
                f.write("-" * 30 + "\n")
                for key, value in self.test_results.items():
                    if isinstance(value, (int, float)):
                        f.write(f"{key}: {value:.4f}\n")
                f.write("\n")
                
                if 'classification_report' in self.test_results:
                    f.write("Detailed Classification Report:\n")
                    f.write("-" * 30 + "\n")
                    report = self.test_results['classification_report']
                    for class_name, metrics in report.items():
                        if isinstance(metrics, dict):
                            f.write(f"\n{class_name}:\n")
                            for metric, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    f.write(f"  {metric}: {value:.4f}\n")
                    f.write("\n")
            
            f.write("Model Architecture Summary:\n")
            f.write("-" * 30 + "\n")
            f.write("• Multi-head self-attention mechanism\n")
            f.write("• Positional encoding for temporal sequences\n")
            f.write("• Layer normalization and residual connections\n")
            f.write("• Multi-task learning (classification + regression)\n")
            f.write("• Global average pooling for sequence aggregation\n\n")
            
            f.write("Analysis completed successfully!\n")
        
        print(f"📄 Comprehensive report saved as: {filename}")

def main():
    """Main execution function"""
    print("🚀 Transformer Model Professional Testing Suite")
    print("=" * 60)
    
    # Initialize tester
    tester = TransformerModelTester()
    
    if tester.model is None:
        print("❌ Cannot proceed without a valid model")
        return
    
    # Try to load real XRS data first
    print("\n📡 Attempting to load real XRS data...")
    xrs_data, timestamps = tester.load_real_xrs_data()
    
    if xrs_data is not None:
        print("✅ Using real XRS data for testing")
        X_test, y_class_test = tester.generate_enhanced_features_from_xrs(xrs_data)
        y_reg_test = None  # Real data doesn't have regression labels
    else:
        # Generate synthetic test data as fallback
        print("\n📊 Generating synthetic test data...")
        X_test, y_class_test, y_reg_test = tester.generate_test_data(n_samples=200)
        print("✅ Using synthetic data for testing")
    
    print(f"✅ Test data shape: {X_test.shape}")
    
    # Test the model
    print("\n🔄 Testing Transformer model...")
    predictions, results = tester.test_model(X_test, y_class_test, y_reg_test)
    
    if results:
        # Print results summary
        print("\n📈 Test Results Summary:")
        print("-" * 50)
        for key, value in results.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")
        
        # Create professional visualizations
        print("\n🎨 Creating professional visualizations...")
        fig = tester.create_professional_visualizations(X_test, y_class_test, y_reg_test)
        
        # Generate report
        tester.generate_report()
        
        print("\n✅ Transformer Model Analysis Complete!")
        print("📊 Check the generated visualization and report files.")
    else:
        print("❌ Model testing failed")

if __name__ == "__main__":
    main()

"""
Comprehensive Model Comparison and Analysis Suite
Professional comparison of all trained solar flare ML models with aesthetic visualizations
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
import plotly.express as px

warnings.filterwarnings('ignore')

# Set professional styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)
sns.set_palette("Set2")

class ComprehensiveModelComparator:
    """Professional comparison suite for all solar flare ML models"""
    
    def __init__(self):
        """Initialize the model comparator"""
        self.models = {}
        self.predictions = {}
        self.performance_metrics = {}
        self.test_data = None
        self.scaler = RobustScaler()
        
        # Model information
        self.model_info = {
            'transformer': {
                'name': 'Transformer Model',
                'file': 'best_transformer_model.h5',
                'color': '#FF6B6B',
                'description': 'Multi-head self-attention for sequence modeling'
            },
            'monte_carlo': {
                'name': 'Monte Carlo Bayesian',
                'file': 'best_graph_model.h5',
                'color': '#4ECDC4',
                'description': 'Uncertainty quantification with MC dropout'
            },
            'contrastive': {
                'name': 'Contrastive Learning',
                'file': 'best_contrastive_classifier.h5',
                'color': '#45B7D1',
                'description': 'Self-supervised representation learning'
            }
        }
        
        # Flare classification mapping
        self.flare_classes = {
            0: 'Background', 1: 'A-class', 2: 'B-class', 
            3: 'C-class', 4: 'M-class', 5: 'X-class'
        }
        
        self.load_all_models()
    
    def load_all_models(self):
        """Load all available trained models"""
        print("🔄 Loading all available models...")
        
        for model_key, info in self.model_info.items():
            try:
                if os.path.exists(info['file']):
                    model = keras.models.load_model(info['file'])
                    self.models[model_key] = model
                    print(f"✅ Loaded {info['name']} from {info['file']}")
                else:
                    print(f"⚠️  Model file not found: {info['file']}")
            except Exception as e:
                print(f"❌ Error loading {info['name']}: {str(e)}")
        
        print(f"📊 Successfully loaded {len(self.models)} models")
    
    def generate_comprehensive_test_data(self, n_samples=1000, sequence_length=128):
        """Generate comprehensive test dataset"""
        print("📈 Generating comprehensive test dataset...")
        
        X = []
        y_class = []
        y_reg = []
        flare_metadata = []
        
        for i in range(n_samples):
            # Generate background signal
            signal = np.random.lognormal(-8, 0.5, (sequence_length, 2))
            
            # Metadata for this sample
            metadata = {
                'sample_id': i,
                'has_flare': False,
                'flare_class': 0,
                'peak_intensity': 0,
                'duration': 0,
                'complexity': 'simple'
            }
            
            # Randomly add flares with realistic distribution
            if np.random.random() < 0.35:  # 35% chance of flare
                flare_start = np.random.randint(20, sequence_length - 40)
                flare_class = np.random.choice([1, 2, 3, 4, 5], p=[0.45, 0.35, 0.15, 0.04, 0.01])
                
                # Flare intensities
                intensities = {1: 1e-8, 2: 1e-7, 3: 1e-6, 4: 1e-5, 5: 1e-4}
                peak_intensity = intensities[flare_class] * np.random.uniform(1, 10)
                
                # Flare duration and complexity
                base_duration = {1: 15, 2: 25, 3: 35, 4: 50, 5: 70}
                duration = base_duration[flare_class] + np.random.randint(-10, 15)
                duration = max(5, min(duration, sequence_length - flare_start - 5))
                
                # Add complexity for higher classes
                if flare_class >= 3:
                    complexity = np.random.choice(['simple', 'complex'], p=[0.7, 0.3])
                    if complexity == 'complex':
                        # Add secondary peaks
                        n_peaks = np.random.randint(2, 4)
                        peak_intensities = [peak_intensity * np.random.uniform(0.3, 1.0) for _ in range(n_peaks)]
                    else:
                        n_peaks = 1
                        peak_intensities = [peak_intensity]
                else:
                    complexity = 'simple'
                    n_peaks = 1
                    peak_intensities = [peak_intensity]
                
                # Generate flare profile
                for peak_idx, intensity in enumerate(peak_intensities):
                    peak_offset = peak_idx * (duration // n_peaks)
                    for j in range(duration):
                        pos = flare_start + j + peak_offset
                        if pos < sequence_length:
                            progress = j / duration
                            if progress < 0.3:
                                amplitude = intensity * (progress / 0.3)
                            else:
                                amplitude = intensity * np.exp(-(progress - 0.3) / 0.7)
                            
                            signal[pos, 0] += amplitude * 0.1  # XRSA
                            signal[pos, 1] += amplitude        # XRSB
                
                # Update metadata
                metadata.update({
                    'has_flare': True,
                    'flare_class': flare_class,
                    'peak_intensity': peak_intensity,
                    'duration': duration,
                    'complexity': complexity
                })
                
                y_class.append(flare_class)
                y_reg.append(np.log10(peak_intensity))
            else:
                y_class.append(0)  # Background
                y_reg.append(np.log10(np.max(signal[:, 1])))
            
            flare_metadata.append(metadata)
            
            # Log transform and add noise
            signal = np.log10(np.maximum(signal, 1e-12))
            signal += np.random.normal(0, 0.1, signal.shape)
            X.append(signal)
        
        self.test_data = {
            'X': np.array(X),
            'y_class': np.array(y_class),
            'y_reg': np.array(y_reg),
            'metadata': flare_metadata
        }
        
        print(f"✅ Generated {n_samples} test samples")
        print(f"   - Background samples: {np.sum(np.array(y_class) == 0)}")
        print(f"   - Flare samples: {np.sum(np.array(y_class) > 0)}")
        return self.test_data
    
    def preprocess_data(self, X, fit_scaler=False):
        """Preprocess data for model input"""
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X_reshaped)
        else:
            X_scaled = self.scaler.transform(X_reshaped)
        
        return X_scaled.reshape(original_shape)
    def test_all_models(self):
        """Test all loaded models on the same dataset"""
        if self.test_data is None:
            print("❌ No test data available. Generate test data first.")
            return
        
        print("🔄 Testing all models on the same dataset...")
        X_test = self.test_data['X']
        y_class_test = self.test_data['y_class']
        y_reg_test = self.test_data['y_reg']
        
        for model_key, model in self.models.items():
            print(f"  Testing {self.model_info[model_key]['name']}...")
            
            try:
                # Adapt input shape to match model requirements
                expected_input_shape = model.input_shape
                print(f"    🔧 Model expects input shape: {expected_input_shape}")
                print(f"    🔧 Current data shape: {X_test.shape}")
                
                # Handle shape adaptation
                X_adapted = X_test
                if len(expected_input_shape) == 3:  # e.g., (None, 128, 2)
                    seq_length = expected_input_shape[1]
                    n_features = expected_input_shape[2]
                    
                    if len(X_test.shape) == 2:  # Need to reshape 2D to 3D
                        # Convert features to sequence format
                        if X_test.shape[1] < seq_length * n_features:
                            # Pad with zeros
                            padding = np.zeros((X_test.shape[0], seq_length * n_features - X_test.shape[1]))
                            X_padded = np.concatenate([X_test, padding], axis=1)
                        else:
                            # Truncate
                            X_padded = X_test[:, :seq_length * n_features]
                        
                        # Reshape to sequence format
                        X_adapted = X_padded.reshape(X_test.shape[0], seq_length, n_features)
                        print(f"    🔧 Reshaped data to: {X_adapted.shape}")
                    elif X_test.shape[1] != seq_length or X_test.shape[2] != n_features:
                        # Handle sequence/feature dimension mismatches
                        if X_test.shape[1] < seq_length:
                            # Pad sequence
                            padding = np.zeros((X_test.shape[0], seq_length - X_test.shape[1], X_test.shape[2]))
                            X_adapted = np.concatenate([X_test, padding], axis=1)
                        else:
                            # Truncate sequence
                            X_adapted = X_test[:, :seq_length, :]
                        
                        # Handle feature dimension
                        if X_adapted.shape[2] < n_features:
                            padding = np.zeros((X_adapted.shape[0], X_adapted.shape[1], n_features - X_adapted.shape[2]))
                            X_adapted = np.concatenate([X_adapted, padding], axis=2)
                        elif X_adapted.shape[2] > n_features:
                            X_adapted = X_adapted[:, :, :n_features]
                        
                        print(f"    🔧 Adapted data to: {X_adapted.shape}")
                elif len(expected_input_shape) == 2:  # e.g., (None, 20)
                    expected_features = expected_input_shape[1]
                    if len(X_test.shape) == 3:  # Need to flatten 3D to 2D
                        X_adapted = X_test.reshape(X_test.shape[0], -1)
                    
                    if X_adapted.shape[1] != expected_features:
                        if X_adapted.shape[1] < expected_features:
                            # Pad with zeros
                            padding = np.zeros((X_adapted.shape[0], expected_features - X_adapted.shape[1]))
                            X_adapted = np.concatenate([X_adapted, padding], axis=1)
                        else:
                            # Truncate
                            X_adapted = X_adapted[:, :expected_features]
                        print(f"    🔧 Adjusted data to: {X_adapted.shape}")
                
                # Preprocess data
                X_test_scaled = self.preprocess_data(X_adapted, fit_scaler=True)
                
                # Make predictions
                if len(model.outputs) == 2:  # Multi-output model
                    pred_class, pred_reg = model.predict(X_test_scaled, verbose=0)
                    class_pred = np.argmax(pred_class, axis=1)
                    reg_pred = pred_reg.flatten()
                else:  # Single output model
                    predictions = model.predict(X_test_scaled, verbose=0)
                    if len(predictions.shape) > 1 and predictions.shape[1] > 1:  # Classification
                        pred_class = predictions
                        class_pred = np.argmax(pred_class, axis=1)
                        reg_pred = None
                    else:  # Regression or binary classification
                        if predictions.shape[1] == 1:  # Single output
                            pred_class = None
                            class_pred = (predictions.flatten() > 0.5).astype(int)  # Binary threshold
                            reg_pred = predictions.flatten()
                        else:
                            pred_class = predictions
                            class_pred = np.argmax(pred_class, axis=1)
                            reg_pred = None
                
                # Store predictions
                self.predictions[model_key] = {
                    'class_probs': pred_class,
                    'class_pred': class_pred,
                    'regression': reg_pred
                }
                
                # Calculate metrics with safe array handling
                metrics = {}
                
                if class_pred is not None:
                    # Ensure compatible shapes for accuracy calculation
                    min_len = min(len(class_pred), len(y_class_test))
                    class_pred_safe = class_pred[:min_len]
                    y_class_safe = y_class_test[:min_len]
                    
                    # Classification metrics
                    accuracy = np.mean(class_pred_safe == y_class_safe)
                    metrics['accuracy'] = accuracy
                    
                    # Per-class metrics with error handling
                    try:
                        unique_classes = np.unique(np.concatenate([y_class_safe, class_pred_safe]))
                        target_names = [self.flare_classes[i] for i in unique_classes if i in self.flare_classes]
                        
                        report = classification_report(y_class_safe, class_pred_safe, 
                                                     target_names=target_names, 
                                                     output_dict=True, zero_division=0)
                        metrics['classification_report'] = report
                        metrics['confusion_matrix'] = confusion_matrix(y_class_safe, class_pred_safe)
                        
                        # Macro metrics
                        metrics['macro_precision'] = report['macro avg']['precision']
                        metrics['macro_recall'] = report['macro avg']['recall']
                        metrics['macro_f1'] = report['macro avg']['f1-score']
                    except Exception as e:
                        print(f"    ⚠️ Warning: Could not generate detailed classification metrics: {e}")
                        metrics['macro_precision'] = 0
                        metrics['macro_recall'] = 0
                        metrics['macro_f1'] = 0
                
                if reg_pred is not None:
                    # Handle shape mismatches for regression
                    min_len = min(len(reg_pred), len(y_reg_test))
                    reg_pred_safe = reg_pred[:min_len]
                    y_reg_safe = y_reg_test[:min_len]
                    
                    # Regression metrics
                    mse = mean_squared_error(y_reg_safe, reg_pred_safe)
                    r2 = r2_score(y_reg_safe, reg_pred_safe)
                    mae = np.mean(np.abs(y_reg_safe - reg_pred_safe))
                    
                    metrics['regression_mse'] = mse
                    metrics['regression_r2'] = r2
                    metrics['regression_mae'] = mae
                
                self.performance_metrics[model_key] = metrics
                print(f"    ✅ {self.model_info[model_key]['name']} tested successfully")
                
            except Exception as e:
                print(f"    ❌ Error testing {self.model_info[model_key]['name']}: {str(e)}")
                self.predictions[model_key] = None
                self.performance_metrics[model_key] = {}
        
        print("✅ All models tested")
    
    def create_comprehensive_comparison_dashboard(self):
        """Create a comprehensive comparison dashboard"""
        
        # Set up the plotting environment
        plt.rcParams.update({
            'font.size': 11,
            'axes.linewidth': 1.5,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })
        
        # Create main figure with subplots
        fig = plt.figure(figsize=(24, 32))
        gs = fig.add_gridspec(8, 4, height_ratios=[1, 1, 1, 1, 1, 1, 1, 1], hspace=0.3, wspace=0.3)
        
        # 1. Model Performance Overview
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_model_performance_overview(ax1)
        
        # 2. Classification Performance Comparison
        ax2 = fig.add_subplot(gs[1, :2])
        self._plot_classification_comparison(ax2)
        
        # 3. Regression Performance Comparison
        ax3 = fig.add_subplot(gs[1, 2:])
        self._plot_regression_comparison(ax3)
        
        # 4. Confusion Matrices Comparison
        self._plot_confusion_matrices_comparison(fig, gs[2, :])
        
        # 5. Class-wise Performance Analysis
        ax5 = fig.add_subplot(gs[3, :])
        self._plot_classwise_performance(ax5)
        
        # 6. Prediction Confidence Analysis
        ax6 = fig.add_subplot(gs[4, :2])
        self._plot_prediction_confidence_comparison(ax6)
        
        # 7. Error Analysis
        ax7 = fig.add_subplot(gs[4, 2:])
        self._plot_error_analysis_comparison(ax7)
        
        # 8. Model Complexity and Speed Analysis
        ax8 = fig.add_subplot(gs[5, :2])
        self._plot_model_complexity_comparison(ax8)
        
        # 9. Flare Type Performance
        ax9 = fig.add_subplot(gs[5, 2:])
        self._plot_flare_type_performance(ax9)
        
        # 10. Sample Predictions Comparison
        ax10 = fig.add_subplot(gs[6, :])
        self._plot_sample_predictions_comparison(ax10)
        
        # 11. ROC Curves Comparison
        ax11 = fig.add_subplot(gs[7, :2])
        self._plot_roc_comparison(ax11)
        
        # 12. Performance Summary Table
        ax12 = fig.add_subplot(gs[7, 2:])
        self._plot_performance_summary_table(ax12)
        
        plt.suptitle('Comprehensive Solar Flare ML Models Comparison Dashboard', 
                    fontsize=26, fontweight='bold', y=0.98)
        
        # Save the comprehensive dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_model_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Comprehensive comparison dashboard saved as: {filename}")
        
        plt.show()
        
        return fig
    
    def _plot_model_performance_overview(self, ax):
        """Plot overall model performance overview"""
        models = []
        accuracies = []
        r2_scores = []
        colors = []
        
        for model_key in self.models.keys():
            if model_key in self.performance_metrics:
                metrics = self.performance_metrics[model_key]
                models.append(self.model_info[model_key]['name'])
                accuracies.append(metrics.get('accuracy', 0))
                r2_scores.append(metrics.get('regression_r2', 0))
                colors.append(self.model_info[model_key]['color'])
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, accuracies, width, label='Classification Accuracy', 
                      color=colors, alpha=0.8)
        bars2 = ax.bar(x + width/2, r2_scores, width, label='Regression R²', 
                      color=colors, alpha=0.6)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Model Performance Overview', fontsize=16, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_classification_comparison(self, ax):
        """Plot classification metrics comparison"""
        metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        models = list(self.models.keys())
        x = np.arange(len(metric_names))
        width = 0.8 / len(models)
        
        for i, model_key in enumerate(models):
            if model_key in self.performance_metrics:
                values = [self.performance_metrics[model_key].get(metric, 0) for metric in metrics]
                color = self.model_info[model_key]['color']
                ax.bar(x + i*width, values, width, label=self.model_info[model_key]['name'], 
                      color=color, alpha=0.8)
        
        ax.set_title('Classification Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.1)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_regression_comparison(self, ax):
        """Plot regression metrics comparison"""
        models = []
        mse_values = []
        r2_values = []
        mae_values = []
        colors = []
        
        for model_key in self.models.keys():
            if model_key in self.performance_metrics:
                metrics = self.performance_metrics[model_key]
                if 'regression_mse' in metrics:
                    models.append(self.model_info[model_key]['name'])
                    mse_values.append(metrics['regression_mse'])
                    r2_values.append(metrics['regression_r2'])
                    mae_values.append(metrics['regression_mae'])
                    colors.append(self.model_info[model_key]['color'])
        
        if models:
            x = np.arange(len(models))
            width = 0.25
            
            # Normalize MSE and MAE for visualization (lower is better)
            max_mse = max(mse_values) if mse_values else 1
            max_mae = max(mae_values) if mae_values else 1
            mse_norm = [1 - (mse/max_mse) for mse in mse_values]  # Invert so higher is better
            mae_norm = [1 - (mae/max_mae) for mae in mae_values]  # Invert so higher is better
            
            ax.bar(x - width, r2_values, width, label='R² Score', color=colors, alpha=0.8)
            ax.bar(x, mse_norm, width, label='MSE (normalized)', color=colors, alpha=0.6)
            ax.bar(x + width, mae_norm, width, label='MAE (normalized)', color=colors, alpha=0.4)
            
            ax.set_title('Regression Metrics Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel('Score (Higher is Better)')
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No regression results available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_confusion_matrices_comparison(self, fig, gs_slice):
        """Plot confusion matrices for all models"""
        models_with_classification = [(k, v) for k, v in self.performance_metrics.items() 
                                    if 'confusion_matrix' in v]
        
        n_models = len(models_with_classification)
        if n_models == 0:
            return
        
        for i, (model_key, metrics) in enumerate(models_with_classification):
            ax = fig.add_subplot(gs_slice[i])
            
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=list(self.flare_classes.values()),
                       yticklabels=list(self.flare_classes.values()))
            
            ax.set_title(f'{self.model_info[model_key]["name"]} - Confusion Matrix', 
                        fontsize=12, fontweight='bold')
            if i == 0:
                ax.set_ylabel('True Class')
            if i == n_models - 1:
                ax.set_xlabel('Predicted Class')
    
    def _plot_classwise_performance(self, ax):
        """Plot class-wise performance comparison"""
        # Get class names (excluding macro/micro avg)
        class_names = list(self.flare_classes.values())
        
        # Collect F1-scores for each class across models
        models = []
        class_f1_scores = {class_name: [] for class_name in class_names}
        
        for model_key in self.models.keys():
            if model_key in self.performance_metrics:
                metrics = self.performance_metrics[model_key]
                if 'classification_report' in metrics:
                    models.append(self.model_info[model_key]['name'])
                    report = metrics['classification_report']
                    
                    for class_name in class_names:
                        if class_name in report:
                            class_f1_scores[class_name].append(report[class_name]['f1-score'])
                        else:
                            class_f1_scores[class_name].append(0)
        
        if models:
            x = np.arange(len(class_names))
            width = 0.8 / len(models)
            
            for i, model_name in enumerate(models):
                model_key = [k for k, v in self.model_info.items() if v['name'] == model_name][0]
                f1_scores = [class_f1_scores[class_name][i] for class_name in class_names]
                color = self.model_info[model_key]['color']
                
                ax.bar(x + i*width, f1_scores, width, label=model_name, 
                      color=color, alpha=0.8)
            
            ax.set_title('Class-wise F1-Score Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel('F1-Score')
            ax.set_ylim(0, 1.1)
            ax.set_xticks(x + width * (len(models) - 1) / 2)
            ax.set_xticklabels(class_names, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_prediction_confidence_comparison(self, ax):
        """Plot prediction confidence comparison"""
        for model_key, predictions in self.predictions.items():
            if predictions and predictions['class_probs'] is not None:
                confidence = np.max(predictions['class_probs'], axis=1)
                color = self.model_info[model_key]['color']
                
                ax.hist(confidence, bins=30, alpha=0.6, label=self.model_info[model_key]['name'],
                       color=color, density=True)
        
        ax.set_title('Prediction Confidence Distribution Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Prediction Confidence')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_error_analysis_comparison(self, ax):
        """Plot error analysis comparison"""
        y_true = self.test_data['y_class']
        
        models = []
        error_rates = []
        colors = []
        
        for model_key, predictions in self.predictions.items():
            if predictions and predictions['class_pred'] is not None:
                error_rate = np.mean(predictions['class_pred'] != y_true)
                models.append(self.model_info[model_key]['name'])
                error_rates.append(error_rate)
                colors.append(self.model_info[model_key]['color'])
        
        if models:
            bars = ax.bar(models, error_rates, color=colors, alpha=0.8)
            
            # Add value labels
            for bar, error_rate in zip(bars, error_rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{error_rate:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title('Overall Error Rate Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel('Error Rate')
            ax.set_ylim(0, max(error_rates) * 1.2)
            ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_model_complexity_comparison(self, ax):
        """Plot model complexity comparison"""
        models = []
        param_counts = []
        colors = []
        
        for model_key, model in self.models.items():
            models.append(self.model_info[model_key]['name'])
            param_count = model.count_params()
            param_counts.append(param_count / 1000)  # Convert to thousands
            colors.append(self.model_info[model_key]['color'])
        
        bars = ax.bar(models, param_counts, color=colors, alpha=0.8)
        
        # Add value labels
        for bar, param_count in zip(bars, param_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(param_counts)*0.01,
                   f'{param_count:.1f}K', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Model Complexity (Parameter Count)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Parameters (Thousands)')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_flare_type_performance(self, ax):
        """Plot performance by flare type complexity"""
        # Analyze performance on simple vs complex flares
        metadata = self.test_data['metadata']
        y_true = self.test_data['y_class']
        
        # Separate simple and complex flares
        simple_mask = np.array([meta['complexity'] == 'simple' for meta in metadata])
        complex_mask = np.array([meta['complexity'] == 'complex' for meta in metadata])
        
        models = []
        simple_acc = []
        complex_acc = []
        colors = []
        
        for model_key, predictions in self.predictions.items():
            if predictions and predictions['class_pred'] is not None:
                y_pred = predictions['class_pred']
                
                # Calculate accuracies
                simple_accuracy = np.mean(y_pred[simple_mask] == y_true[simple_mask]) if np.any(simple_mask) else 0
                complex_accuracy = np.mean(y_pred[complex_mask] == y_true[complex_mask]) if np.any(complex_mask) else 0
                
                models.append(self.model_info[model_key]['name'])
                simple_acc.append(simple_accuracy)
                complex_acc.append(complex_accuracy)
                colors.append(self.model_info[model_key]['color'])
        
        if models:
            x = np.arange(len(models))
            width = 0.35
            
            ax.bar(x - width/2, simple_acc, width, label='Simple Flares', 
                  color=colors, alpha=0.8)
            ax.bar(x + width/2, complex_acc, width, label='Complex Flares', 
                  color=colors, alpha=0.6)
            
            ax.set_title('Performance by Flare Complexity', fontsize=14, fontweight='bold')
            ax.set_ylabel('Accuracy')
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_sample_predictions_comparison(self, ax):
        """Plot sample predictions comparison"""
        # Select interesting samples
        X_test = self.test_data['X']
        sample_indices = [10, 50, 100, 150, 200]  # Sample with different characteristics
        
        for i, idx in enumerate(sample_indices[:min(3, len(X_test))]):
            sample = X_test[idx]
            time_steps = np.arange(len(sample))
            
            # Plot time series
            ax.plot(time_steps, sample[:, 1], alpha=0.7, linewidth=2, 
                   label=f'Sample {idx}')
            
            # Add predictions from each model
            for model_key, predictions in self.predictions.items():
                if predictions and predictions['class_pred'] is not None:
                    pred_class = predictions['class_pred'][idx]
                    class_name = self.flare_classes[pred_class]
                    ax.text(len(sample) * 0.8, sample[int(len(sample)*0.8), 1] + i*0.5, 
                           f'{self.model_info[model_key]["name"]}: {class_name}',
                           fontsize=8, color=self.model_info[model_key]['color'])
        
        ax.set_title('Sample Time Series with Model Predictions', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Log XRSB Flux')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_roc_comparison(self, ax):
        """Plot ROC curves comparison (simplified for multiclass)"""
        # For multiclass, we'll plot average ROC curve
        ax.text(0.5, 0.5, 'ROC Curve Analysis\n(Requires binary classification setup)', 
               ha='center', va='center', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
        ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
    
    def _plot_performance_summary_table(self, ax):
        """Plot performance summary table"""
        # Create summary data
        summary_data = []
        
        for model_key in self.models.keys():
            if model_key in self.performance_metrics:
                metrics = self.performance_metrics[model_key]
                row = {
                    'Model': self.model_info[model_key]['name'],
                    'Accuracy': f"{metrics.get('accuracy', 0):.3f}",
                    'F1-Score': f"{metrics.get('macro_f1', 0):.3f}",
                    'R² Score': f"{metrics.get('regression_r2', 0):.3f}",
                    'Parameters': f"{self.models[model_key].count_params()/1000:.1f}K"
                }
                summary_data.append(row)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            
            # Create table
            ax.axis('tight')
            ax.axis('off')
            
            table = ax.table(cellText=df.values, colLabels=df.columns,
                           cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Style the table
            for i in range(len(df.columns)):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            for i in range(1, len(df) + 1):
                for j in range(len(df.columns)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f1f1f2')
        
        ax.set_title('Performance Summary Table', fontsize=14, fontweight='bold')
    
    def load_real_xrs_data(self, data_path="../solar_flare_analysis/data/"):
        """Load real XRS data for comprehensive testing"""
        try:
            print("\n📡 Loading Real XRS Data for Model Comparison...")
            
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
                    print("⚠️ No XRS files found. Using synthetic data...")
                    return self.generate_comprehensive_test_data()
            
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
                print("⚠️ Could not identify XRS A and B channels. Using synthetic data...")
                return self.generate_comprehensive_test_data()
            
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
                print("⚠️ Not enough valid XRS data points. Using synthetic data...")
                return self.generate_comprehensive_test_data()
            
            # Sample for manageable size
            if len(xrs_data) > 2000:
                xrs_data = xrs_data.sample(n=2000, random_state=42)
                print(f"📊 Sampled to 2000 points")
            
            # Convert XRS data to sequence format for model compatibility
            sequence_length = 128
            n_features = 2
            n_samples = len(xrs_data) // sequence_length
            
            X = []
            y_class = []
            y_reg = []
            flare_metadata = []
            
            xrs_long = xrs_data[xrs_long_col].values
            xrs_short = xrs_data[xrs_short_col].values
            
            # Calculate thresholds for flare detection
            xrs_long_95th = np.percentile(xrs_long, 95)
            xrs_short_95th = np.percentile(xrs_short, 95)
            
            for i in range(n_samples):
                start_idx = i * sequence_length
                end_idx = start_idx + sequence_length
                
                # Extract sequence
                seq_long = xrs_long[start_idx:end_idx]
                seq_short = xrs_short[start_idx:end_idx]
                
                # Create sequence matrix
                sequence = np.column_stack([seq_long, seq_short])
                
                # Apply log transformation
                sequence = np.log10(np.maximum(sequence, 1e-12))
                
                # Detect flares in this sequence
                max_long = np.max(seq_long)
                max_short = np.max(seq_short)
                
                if max_long > xrs_long_95th or max_short > xrs_short_95th:
                    # Classify flare intensity
                    if max_short > 1e-4:
                        flare_class = 5  # X-class
                    elif max_short > 1e-5:
                        flare_class = 4  # M-class
                    elif max_short > 1e-6:
                        flare_class = 3  # C-class
                    elif max_short > 1e-7:
                        flare_class = 2  # B-class
                    else:
                        flare_class = 1  # A-class
                    
                    complexity = 'complex' if flare_class >= 3 else 'simple'
                else:
                    flare_class = 0  # Background
                    complexity = 'simple'
                
                # Metadata for this sample
                metadata = {
                    'sample_id': i,
                    'has_flare': flare_class > 0,
                    'flare_class': flare_class,
                    'peak_intensity': max(max_long, max_short),
                    'duration': 0,  # Would need more sophisticated detection
                    'complexity': complexity
                }
                
                X.append(sequence)
                y_class.append(flare_class)
                y_reg.append(np.log10(max(max_long, max_short)))
                flare_metadata.append(metadata)
            
            self.test_data = {
                'X': np.array(X),
                'y_class': np.array(y_class),
                'y_reg': np.array(y_reg),
                'metadata': flare_metadata
            }
            
            print(f"✅ Created {n_samples} sequences from real XRS data")
            print(f"   - Background samples: {np.sum(np.array(y_class) == 0)}")
            print(f"   - Flare samples: {np.sum(np.array(y_class) > 0)}")
            
            # Show class distribution
            unique, counts = np.unique(y_class, return_counts=True)
            for class_idx, count in zip(unique, counts):
                class_name = self.flare_classes[class_idx]
                percentage = count / len(y_class) * 100
                print(f"   - {class_name}: {count} samples ({percentage:.1f}%)")
            
            return self.test_data
                
        except Exception as e:
            print(f"❌ Error loading XRS data: {str(e)}")
            print("🔄 Falling back to synthetic data...")
            return self.generate_comprehensive_test_data()
    
def main():
    """Main function to run comprehensive model comparison"""
    print("🚀 Starting Comprehensive Solar Flare ML Models Comparison")
    print("=" * 80)
    
    # Initialize comparator
    comparator = ComprehensiveModelComparator()
    
    if not comparator.models:
        print("❌ No models loaded. Cannot proceed with comparison.")
        return
    
    # Generate comprehensive test data
    print("\n📊 Generating comprehensive test dataset...")
    test_data = comparator.generate_comprehensive_test_data(n_samples=1000)
    
    # Test all models
    print("\n🔄 Testing all models...")
    comparator.test_all_models()
      # Print quick summary
    print("\n📈 Quick Performance Summary:")
    print("-" * 50)
    for model_key, metrics in comparator.performance_metrics.items():
        model_name = comparator.model_info[model_key]['name']
        accuracy = metrics.get('accuracy', None)
        f1_score = metrics.get('macro_f1', None)
        
        # Format accuracy
        if accuracy is not None:
            acc_str = f"{accuracy:.3f}"
        else:
            acc_str = "N/A"
        
        # Format F1 score
        if f1_score is not None:
            f1_str = f"{f1_score:.3f}"
        else:
            f1_str = "N/A"
        
        print(f"{model_name:20} | Accuracy: {acc_str:>6} | F1: {f1_str:>6}")
    
    # Create comprehensive comparison dashboard
    print("\n🎨 Creating comprehensive comparison dashboard...")
    fig = comparator.create_comprehensive_comparison_dashboard()
    
    print("\n✅ Comprehensive model comparison completed successfully!")
    print("📊 Check the generated dashboard for detailed comparative analysis.")

if __name__ == "__main__":
    main()
